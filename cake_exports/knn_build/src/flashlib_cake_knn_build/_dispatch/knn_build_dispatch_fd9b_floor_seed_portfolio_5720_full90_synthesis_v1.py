"""Full90 synthesis over fd9b for floor-clearing seed portfolio routes.

Minimum target architecture: sm_100a. This dispatcher-synthesis wrapper keeps
the fd9b full90 champion as the baseline fallback and measures guarded
Weave-only portfolios over the post-trunk seed bank:

* trunk K20 + K32 exact build seeds from bd76 and 9334;
* 01bb Q4096/K8, 2425 RAG micro K10 q4/q8/q32, and K20/K32;
* the same portfolio plus the read-ref 1b34/FCEE FlashML K5 seed.

The wrapper does not retune seed schedules and does not change the default
knn_build registry route. K48 remains parked because d047/5698 showed aggregate
regression. FlashLib is timed only by the contract harness.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1 as fd9b
from . import knn_build_flashml_k5_bd4a_v1 as flashml_k5
from . import knn_build_large_square_k20_efe4_v1 as k20_bd76
from . import knn_build_overk_largek_q4096_k32_9334_v1 as k32_9334
from . import knn_build_q4096_k8_lowfloor_fd9b_v3 as q4096_01bb
from . import knn_build_rag_microbatch_k10_q4q8q32_s144_d5ac_v1 as rag_2425
MODULE = 'loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1'
eval_mod = fd9b.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
BASE_CANDIDATE_KEY = fd9b.DEFAULT_CANDIDATE_KEY
BASE_CONFIG = fd9b.CANDIDATE_CONFIGS[BASE_CANDIDATE_KEY]
BASE_CANDIDATE_ID = BASE_CONFIG['candidate_id']
BASE_BENCHMARK_ENTRYPOINT = BASE_CONFIG['benchmark_entrypoint']
BASE_ROUTE_ENTRYPOINT = BASE_CONFIG['entrypoint']
CANDIDATE_K20_K32 = 'fd9b_plus_bd76_k20_9334_k32'
CANDIDATE_K5_ONLY = 'fd9b_plus_1b34_k5'
CANDIDATE_FLOOR_CORE = 'fd9b_plus_01bb_2425_bd76_k20_9334_k32'
CANDIDATE_FLOOR_CORE_K5 = 'fd9b_plus_01bb_2425_1b34_k5_bd76_k20_9334_k32'
DEFAULT_CANDIDATE_KEY = CANDIDATE_FLOOR_CORE_K5
CANDIDATE_KEYS = (BASE_CANDIDATE_KEY, CANDIDATE_K20_K32, CANDIDATE_K5_ONLY, CANDIDATE_FLOOR_CORE, CANDIDATE_FLOOR_CORE_K5)
DEFAULT_SYNTHESIS_CANDIDATES = (CANDIDATE_K20_K32, CANDIDATE_K5_ONLY, CANDIDATE_FLOOR_CORE, CANDIDATE_FLOOR_CORE_K5)
FULL90_LABELS = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_k_sweep_qm1024_k16", "build_k_sweep_qm1024_k12", "build_k_sweep_qm1024_k20", "build_qm2048_d128_k8", "build_qm1024_d128_k8", "build_qm4096_d128_k8", "build_qm2048_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_common_d256_b1_q1024_m1024_k10", "build_common_d768_b1_q1024_m1024_k10", "build_common_d1024_b1_q512_m512_k10", "build_common_d4096_b1_q512_m512_k10", "build_highd_b1_q1024_m1024_d320_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k12", "build_k_sweep_qm2048_k13", "build_k_sweep_qm2048_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_tail_b1_q1536_m1536_d128_k10", "build_tail_b1_q3072_m3072_d128_k20", "build_medium_b1_q4096_m4096_d128_k10", "build_k_sweep_qm4096_k12", "build_k_sweep_qm4096_k13", "build_k_sweep_qm4096_k20", "build_k_sweep_qm4096_k24", "build_k_sweep_qm4096_k28", "build_largek_stress_qm4096_k32", "build_k_sweep_qm4096_k30", "build_over32_stress_qm2048_k48", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k48", "build_large_b1_q8192_m8192_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_verylarge_b1_q12288_m12288_d128_k10", "rag_offline_b1_q4096_m100000_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "search_rect_b1_q1024_m32768_d64_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "search_rect_common_d256_b1_q1024_m32768_k10", "search_rect_common_d768_b1_q512_m8192_k10", "search_rect_b1_q4096_m65536_d128_k20", "search_rect_b1_q1536_m65536_d128_k20", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_stream_b1_q128_m100000_d128_k10", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_offline_batch_b1_q10000_m100000_d128_k10", "rag_offline_b1_q10000_m50000_d128_k10", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "rag_microbatch_common_d64_b1_q16_m50000_k10", "rag_microbatch_common_d256_b1_q16_m50000_k10", "rag_microbatch_common_d1024_b1_q8_m50000_k10", "rag_microbatch_common_d4096_b1_q4_m32768_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_b1_q64_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_over32_stress_qm4096_k64", "build_over64_stress_qm1024_k96", "build_over64_stress_qm2048_k96", "build_over64_stress_qm4096_k96", "rag_online_common_d64_b1_q1_m262143_k10", "rag_microbatch_common_d64_b1_q4_m100000_k10", "rag_microbatch_common_d256_b1_q4_m100000_k10", "rag_stream_common_d256_b1_q128_m100000_k10", "rag_microbatch_common_d768_b1_q8_m100000_k10", "rag_microbatch_common_d1024_b1_q4_m100000_k10", "rag_online_common_d4096_b1_q1_m65536_k10", "search_rect_common_d1024_b1_q256_m8192_k10", "search_rect_common_d4096_b1_q128_m4096_k10", "rag_microbatch_largek_common_d256_b1_q8_m100000_k32", "rag_stream_largek_common_d256_b1_q128_m100000_k32", "rag_microbatch_over32_d128_b1_q16_m100000_k48"]}'))
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BASELINE_BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_baseline_fd9b_full90_v1'])
K20_K32_BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_k20_k32_full90_v1'])
K5_ONLY_BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_k5_only_full90_v1'])
FLOOR_CORE_BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_floor_core_full90_v1'])
FLOOR_CORE_K5_BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_floor_core_k5_full90_v1'])
K20_TARGET_SHAPES = k20_bd76.TARGET_SHAPES
K32_TARGET_SHAPES = k32_9334.TARGET_SHAPES
Q4096_K8_TARGET_SHAPES = q4096_01bb.TARGET_SHAPES
RAG_2425_TARGET_SHAPES = rag_2425.TARGET_SHAPES
K5_TARGET_SHAPES = (flashml_k5.TARGET_SHAPE,)
K48_EXCLUDED_TARGET_SHAPES = ('build_over32_stress_qm2048_k48', 'build_over32_stress_qm4096_k48')
K20_K32_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_largek_stress_qm4096_k32"]}'))
FLOOR_CORE_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm4096_d128_k8", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_largek_stress_qm4096_k32"]}'))
FLOOR_CORE_K5_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm4096_d128_k8", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_largek_stress_qm4096_k32", "flashml_correctness_b1_q256_m256_d128_k5"]}'))
SEED_BD76_K20_ID = 'large_square_k20_efe4_bd76_split2_warp8'
SEED_9334_K32_ID = k32_9334.SEED_ID
SEED_01BB_Q4096_K8_ID = q4096_01bb.SEED_ID
SEED_2425_RAG_Q4_ID = rag_2425.Q4_S144_SEED_ID
SEED_2425_RAG_Q8_ID = rag_2425.Q8_S144_SEED_ID
SEED_2425_RAG_Q32_ID = rag_2425.parent_exact5.Q32_SEED_ID
SEED_1B34_K5_ID = 'fcee_flashml_k5_bd4a_exact'
SEED_E087_K48_ID = 'over32_prefill_efe4_replay_d047_k48_s4'
SEED_28D8_Q4096_K8_ID = 'q4096_k8_lowfloor_fd9b_v3_split4_28d8'
K20_ENTRYPOINT = 'loom.examples.weave.knn_build_large_square_k20_efe4_v1:launch_from_contract_inputs'
K32_ENTRYPOINT = k32_9334.ROUTE_ENTRYPOINT
Q4096_K8_ENTRYPOINT = q4096_01bb.ROUTE_ENTRYPOINT
RAG_2425_ENTRYPOINT = rag_2425.ROUTE_ENTRYPOINT
K5_ENTRYPOINT = 'loom.examples.weave.knn_build_flashml_k5_bd4a_v1:launch_from_contract_inputs'
SOURCE_TASKS = {**fd9b.SOURCE_TASKS, SEED_BD76_K20_ID: 'weave-evolve-knn-build-bd76 / design_doc/active/weave_evolve_knn_build_round_157_efe4.md', SEED_9334_K32_ID: 'weave-evolve-knn-build-9334 / design_doc/active/weave_evolve_knn_build_round_157_9334.md', SEED_01BB_Q4096_K8_ID: 'weave-evolve-knn-build-01bb / design_doc/active/weave_evolve_knn_build_round_158_fd9b_q4096k8.md', SEED_2425_RAG_Q4_ID: 'weave-evolve-knn-build-2425 / design_doc/active/weave_evolve_knn_build_round_157_d5ac.md', SEED_2425_RAG_Q8_ID: 'weave-evolve-knn-build-2425 / design_doc/active/weave_evolve_knn_build_round_157_d5ac.md', SEED_2425_RAG_Q32_ID: 'weave-evolve-knn-build-2425 parent exact-five q32 route / design_doc/active/weave_evolve_knn_build_round_157_d5ac.md', SEED_1B34_K5_ID: 'weave-evolve-knn-build-1b34 read-ref / design_doc/active/weave_evolve_knn_build_round_149_fcee_flashmlk5.md', SEED_E087_K48_ID: 'generalize-auto-tuning-knn-build-d047 replay evidence only', SEED_28D8_Q4096_K8_ID: 'weave-evolve-knn-build-28d8 / superseded by weave-evolve-knn-build-01bb'}
PRODUCTION_ROUTE_MODULES = {**fd9b.PRODUCTION_ROUTE_MODULES, SEED_BD76_K20_ID: K20_ENTRYPOINT, SEED_9334_K32_ID: K32_ENTRYPOINT, SEED_01BB_Q4096_K8_ID: Q4096_K8_ENTRYPOINT, SEED_2425_RAG_Q4_ID: RAG_2425_ENTRYPOINT, SEED_2425_RAG_Q8_ID: RAG_2425_ENTRYPOINT, SEED_2425_RAG_Q32_ID: RAG_2425_ENTRYPOINT, SEED_1B34_K5_ID: K5_ENTRYPOINT, BASE_CANDIDATE_ID: BASE_ROUTE_ENTRYPOINT}
TARGETED_SEED_ROWS = _decode_capture(_json_loads('{"__dict_items__": [["large_square_k20_efe4_bd76_split2_warp8", {"__dict_items__": [["source_payload", "design_doc/active/weave_evolve_knn_build_round_157_efe4.md#perf"], ["shape_labels", {"__tuple__": ["build_large_b1_q8192_m8192_d128_k20"]}], ["source_task", "weave-evolve-knn-build-bd76"], ["historical_ratio_vs_flashlib", 1.528100456551165]]}], ["overk_largek_q4096_k32_9334_v1", {"__dict_items__": [["source_payload", "artifacts/weave_evolve/knn_build_overk_largek_q4096_k32_9334_v1/largek_q4096_k32_9334_v1.json"], ["shape_labels", {"__tuple__": ["build_largek_stress_qm4096_k32"]}], ["source_task", "weave-evolve-knn-build-9334"], ["historical_ratio_vs_flashlib", 1.482571234983885]]}], ["q4096_k8_lowfloor_fd9b_v3_exact_prefill_s4", {"__dict_items__": [["source_payload", "artifacts/weave_evolve/knn_build_q4096_k8_lowfloor_fd9b_v3/q4096_k8_lowfloor_fd9b_v3.json"], ["shape_labels", {"__tuple__": ["build_qm4096_d128_k8"]}], ["source_task", "weave-evolve-knn-build-01bb"], ["historical_ratio_vs_flashlib", 1.23252]]}], ["rag_microbatch_k10_q4_s144_g12_d555_v1", {"__dict_items__": [["source_payload", "artifacts/weave_evolve/knn_build_ragmicro_q4q8q32_s144_d5ac_v1/rag_microbatch_k10_q4q8_s144_d5ac_v1.json"], ["shape_labels", {"__tuple__": ["rag_microbatch_b1_q4_m100000_d128_k10"]}], ["source_task", "weave-evolve-knn-build-2425"], ["historical_ratio_vs_flashlib", "floor-clearing exact-three payload"]]}], ["rag_microbatch_k10_q8_s144_g12_d5ac_v1", {"__dict_items__": [["source_payload", "artifacts/weave_evolve/knn_build_ragmicro_q4q8q32_s144_d5ac_v1/rag_microbatch_k10_q4q8_s144_d5ac_v1.json"], ["shape_labels", {"__tuple__": ["rag_microbatch_b1_q8_m100000_d128_k10"]}], ["source_task", "weave-evolve-knn-build-2425"], ["historical_ratio_vs_flashlib", "floor-clearing exact-three payload"]]}], ["rag_microbatch_k10_q32_m64_s128_g8_4757_v1", {"__dict_items__": [["source_payload", "artifacts/weave_evolve/knn_build_ragmicro_q4q8q32_s144_d5ac_v1/rag_microbatch_k10_q4q8_s144_d5ac_v1.json"], ["shape_labels", {"__tuple__": ["rag_microbatch_b1_q32_m100000_d128_k10"]}], ["source_task", "weave-evolve-knn-build-2425"], ["historical_ratio_vs_flashlib", "floor-clearing exact-three payload"]]}], ["fcee_flashml_k5_bd4a_exact", {"__dict_items__": [["source_payload", "artifacts/weave_evolve/knn_build_current_champion_flashml_k5_fcee_v1/current_champion_flashml_k5_fcee_v1.json"], ["shape_labels", {"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5"]}], ["source_task", "weave-evolve-knn-build-1b34"], ["historical_ratio_vs_flashlib", 2.504516129032258]]}], ["over32_prefill_efe4_replay_d047_k48_s4", {"__dict_items__": [["source_payload", "sibling weave-evolve-knn-build-e087/design_doc/active/weave_evolve_knn_build_round_157_efe4.md#perf"], ["shape_labels", {"__tuple__": ["build_over32_stress_qm2048_k48", "build_over32_stress_qm4096_k48"]}], ["source_task", "weave-evolve-knn-build-e087"], ["historical_ratio_vs_flashlib", {"__dict_items__": [["build_over32_stress_qm2048_k48", 1.3080669710806698], ["build_over32_stress_qm4096_k48", 1.2405164867230813]]}], ["consumption_status", "parked_aggregate_regression_in_d047_and_5698"]]}], ["q4096_k8_lowfloor_fd9b_v3_split4_28d8", {"__dict_items__": [["source_payload", "artifacts/weave_evolve/knn_build_lowk_q4096_floor_28d8_v1/lowk_q4096_floor_28d8_v1.json"], ["shape_labels", {"__tuple__": ["build_qm4096_d128_k8"]}], ["source_task", "weave-evolve-knn-build-28d8"], ["historical_ratio_vs_flashlib", 1.097558072461247], ["consumption_status", "dominated_by_01bb"]]}]]}'))
REJECTED_ROUTE_COMBINATIONS = ({'id': '28d8_q4096_k8_lowfloor_not_consumed', 'entrypoint': 'loom.examples.weave.knn_build_lowk_q4096_floor_28d8_v1:launch_from_contract_inputs', 'status': 'dominated_by_01bb', 'source_task': 'weave-evolve-knn-build-28d8', 'reason': 'Q4096/K8 28d8/89c2/f53e variants are below floor or dominated; 01bb clears 1.20x FlashLib.'}, {'id': 'e087_k48_not_consumed', 'entrypoint': 'loom.examples.weave.knn_build_over32_prefill_efe4_replay_d047_v1:launch_from_contract_inputs', 'status': 'parked_aggregate_regression', 'source_task': 'weave-evolve-knn-build-e087', 'reason': 'd047/5698 show K48 exact rows clear locally but regress full90 aggregate when added.'})

def _select_contract_shapes(shape_labels):
    return fd9b._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return fd9b._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return fd9b._normalize_route_row(row)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _timing_backend_name(use_cupti: bool) -> str:
    return fd9b._timing_backend_name(use_cupti)

def _payload_shape_labels(shape_labels) -> str | tuple[str, ...]:
    if shape_labels is None:
        return 'full90_v10'
    return tuple((str(label) for label in shape_labels))

def _shape_labels(shape_labels) -> tuple[str, ...]:
    if shape_labels is None:
        return FULL90_LABELS
    return tuple((str(label) for label in shape_labels))

def _denominator_name(shape_labels) -> str:
    labels = _shape_labels(shape_labels)
    if labels == FULL90_LABELS:
        return 'full90_v10'
    if labels == K20_K32_TARGET_SHAPES:
        return 'bd76_k20_9334_k32_exact2'
    if labels == K5_TARGET_SHAPES:
        return '1b34_k5_exact1'
    if labels == FLOOR_CORE_TARGET_SHAPES:
        return '01bb_2425_bd76_9334_exact6'
    if labels == FLOOR_CORE_K5_TARGET_SHAPES:
        return '01bb_2425_1b34_bd76_9334_exact7'
    return ''.join(['custom_', format(len(labels), '')])

def _denominator_label(shape_labels) -> str:
    return 'full90' if _denominator_name(shape_labels) == 'full90_v10' else _denominator_name(shape_labels)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for report in reports:
        for row in report.get('per_shape', {}).values():
            backend = row.get('timing_backend')
            if backend:
                values.append(str(backend))
    return sorted(set(values))

def _selected_seeds(*seed_groups: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for group in seed_groups:
        values.extend(group)
    return tuple(dict.fromkeys(values))

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown fd9b floor-seed portfolio candidate ', format(repr(candidate_key), '')])) from exc

def _candidate_id(candidate_key: str | None) -> str | None:
    if candidate_key is None:
        return None
    return str(_candidate_config(candidate_key)['candidate_id'])

def _candidate_has_k20_k32(candidate_key: str) -> bool:
    return candidate_key in (CANDIDATE_K20_K32, CANDIDATE_FLOOR_CORE, CANDIDATE_FLOOR_CORE_K5)

def _candidate_has_floor_core(candidate_key: str) -> bool:
    return candidate_key in (CANDIDATE_FLOOR_CORE, CANDIDATE_FLOOR_CORE_K5)

def _candidate_has_k5(candidate_key: str) -> bool:
    return candidate_key in (CANDIDATE_K5_ONLY, CANDIDATE_FLOOR_CORE_K5)

def _label_can_hit(inputs: dict[str, Any], labels: tuple[str, ...]) -> bool:
    value = inputs.get('label')
    return value is None or str(value) in labels

def _eligible_rag_2425(inputs: dict[str, Any]) -> bool:
    if not _label_can_hit(inputs, RAG_2425_TARGET_SHAPES):
        return False
    return not bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) in (4, 8, 32)) and (int(inputs.get('M', -1)) == 100000) and (int(inputs.get('D', -1)) == 128) and (int(inputs.get('K', -1)) == 10) and (str(inputs.get('dtype', 'bfloat16')).replace('torch.', '') == 'bfloat16')

def _eligible_flashml_k5(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, K5_TARGET_SHAPES) and flashml_k5._is_target_shape(inputs)

def _matched_seed(inputs: dict[str, Any], candidate_key: str) -> str | None:
    if _candidate_has_floor_core(candidate_key) and q4096_01bb._eligible_q4096_k8(inputs):
        return SEED_01BB_Q4096_K8_ID
    if _candidate_has_floor_core(candidate_key) and _eligible_rag_2425(inputs):
        return rag_2425._selected_seed(inputs)
    if _candidate_has_k5(candidate_key) and _eligible_flashml_k5(inputs):
        return SEED_1B34_K5_ID
    if _candidate_has_k20_k32(candidate_key) and k20_bd76._eligible_q8192_k20(inputs):
        return SEED_BD76_K20_ID
    if _candidate_has_k20_k32(candidate_key) and k32_9334._eligible_q4096_k32(inputs):
        return SEED_9334_K32_ID
    return None

def _seed_route(seed_id: str, inputs: dict[str, Any]) -> str:
    if seed_id == SEED_BD76_K20_ID:
        return k20_bd76.ROUTE_Q8192_K20_SPLIT2
    if seed_id == SEED_9334_K32_ID:
        return k32_9334.route_for_contract_inputs(inputs)
    if seed_id == SEED_01BB_Q4096_K8_ID:
        return q4096_01bb.route_for_contract_inputs(inputs)
    if seed_id in (SEED_2425_RAG_Q4_ID, SEED_2425_RAG_Q8_ID, SEED_2425_RAG_Q32_ID):
        return rag_2425.route_for_contract_inputs(inputs)
    if seed_id == SEED_1B34_K5_ID:
        return K5_ENTRYPOINT
    raise ValueError(''.join(['unknown seed id ', format(repr(seed_id), '')]))

def _seed_entrypoint(seed_id: str) -> str:
    if seed_id == SEED_BD76_K20_ID:
        return K20_ENTRYPOINT
    if seed_id == SEED_9334_K32_ID:
        return K32_ENTRYPOINT
    if seed_id == SEED_01BB_Q4096_K8_ID:
        return Q4096_K8_ENTRYPOINT
    if seed_id in (SEED_2425_RAG_Q4_ID, SEED_2425_RAG_Q8_ID, SEED_2425_RAG_Q32_ID):
        return RAG_2425_ENTRYPOINT
    if seed_id == SEED_1B34_K5_ID:
        return K5_ENTRYPOINT
    raise ValueError(''.join(['unknown seed id ', format(repr(seed_id), '')]))

def _seed_launch(seed_id: str, inputs: dict[str, Any]) -> None:
    if seed_id == SEED_BD76_K20_ID:
        k20_bd76.launch_from_contract_inputs(inputs)
        return
    if seed_id == SEED_9334_K32_ID:
        k32_9334.launch_from_contract_inputs(inputs)
        return
    if seed_id == SEED_01BB_Q4096_K8_ID:
        q4096_01bb.launch_from_contract_inputs(inputs)
        return
    if seed_id in (SEED_2425_RAG_Q4_ID, SEED_2425_RAG_Q8_ID, SEED_2425_RAG_Q32_ID):
        rag_2425.launch_from_contract_inputs(inputs)
        return
    if seed_id == SEED_1B34_K5_ID:
        flashml_k5.launch_from_contract_inputs(inputs)
        return
    raise ValueError(''.join(['unknown seed id ', format(repr(seed_id), '')]))

def _seed_guard(seed_id: str) -> tuple[str, str]:
    if seed_id == SEED_BD76_K20_ID:
        return ('bd76_large_square_k20_split2_warp8_guard', 'exact BF16 build B=1 Q=M=8192 D=128 K=20 split2 warp8')
    if seed_id == SEED_9334_K32_ID:
        return ('9334_largek_q4096_k32_split4_warpselect_guard', 'exact BF16 build B=1 Q=M=4096 D=128 K=32 split4 unordered warp-select')
    if seed_id == SEED_01BB_Q4096_K8_ID:
        return ('01bb_q4096_k8_exact_prefill_s4_guard', 'exact BF16 build B=1 Q=M=4096 D=128 K=8 split4 exact-prefill warp-select')
    if seed_id == SEED_2425_RAG_Q4_ID:
        return ('2425_rag_microbatch_k10_q4_s144_guard', 'exact BF16 non-build B=1 Q=4 M=100000 D=128 K=10 S144/G12')
    if seed_id == SEED_2425_RAG_Q8_ID:
        return ('2425_rag_microbatch_k10_q8_s144_guard', 'exact BF16 non-build B=1 Q=8 M=100000 D=128 K=10 S144/G12')
    if seed_id == SEED_2425_RAG_Q32_ID:
        return ('2425_rag_microbatch_k10_q32_parent_guard', 'exact BF16 non-build B=1 Q=32 M=100000 D=128 K=10 parent M64 route')
    if seed_id == SEED_1B34_K5_ID:
        return ('1b34_flashml_k5_fcee_guard', 'exact BF16 build B=1 Q=M=256 D=128 K=5')
    raise ValueError(''.join(['unknown seed id ', format(repr(seed_id), '')]))

def _base_route(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return fd9b.route_for_contract_inputs(inputs, candidate_key=BASE_CANDIDATE_KEY, force_fallback=force_fallback)

def _base_launch(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    fd9b.launch_from_contract_inputs(inputs, candidate_key=BASE_CANDIDATE_KEY, force_fallback=force_fallback)

def _base_trace_row(label: str, *, force_fallback: bool=False) -> dict[str, Any]:
    return dict(fd9b.route_trace_for_contract_shapes((label,), candidate_key=BASE_CANDIDATE_KEY, force_fallback=force_fallback)[0])

def _expected_seed(inputs: dict[str, Any], candidate_key: str) -> str | None:
    seed_id = _matched_seed(inputs, candidate_key)
    if seed_id is not None:
        return seed_id
    return fd9b._expected_seed(inputs, BASE_CANDIDATE_KEY)
PARENT_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge"]}'))
K20_K32_SEEDS = (SEED_BD76_K20_ID, SEED_9334_K32_ID)
FLOOR_CORE_SEEDS = (SEED_01BB_Q4096_K8_ID, SEED_2425_RAG_Q4_ID, SEED_2425_RAG_Q8_ID, SEED_2425_RAG_Q32_ID, *K20_K32_SEEDS)
K5_ONLY_SEEDS = (SEED_1B34_K5_ID,)
FLOOR_CORE_K5_SEEDS = (*FLOOR_CORE_SEEDS, SEED_1B34_K5_ID)
CANDIDATE_CONFIGS = _decode_capture(_json_loads('{"__dict_items__": [["1877_plus_9a17_fp16_fd37_lowfloor", {"__dict_items__": [["candidate_id", "candidate_1877_plus_9a17_fp16_fd37_lowfloor_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:benchmark_candidate_fp16_fd37_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge"]}], ["new_seed_ids", {"__tuple__": []}], ["guard_plan", {"__tuple__": ["fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session fd9b full90 champion baseline"]]}], ["fd9b_plus_bd76_k20_9334_k32", {"__dict_items__": [["candidate_id", "candidate_fd9b_plus_bd76_k20_9334_k32_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:candidate_k20_k32_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:benchmark_candidate_k20_k32_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge", "large_square_k20_efe4_bd76_split2_warp8", "overk_largek_q4096_k32_9334_v1"]}], ["new_seed_ids", {"__tuple__": ["large_square_k20_efe4_bd76_split2_warp8", "overk_largek_q4096_k32_9334_v1"]}], ["guard_plan", {"__tuple__": ["bd76 exact BF16 build B=1 Q=M=8192 D=128 K=20 split2 warp8 guard", "9334 exact BF16 build B=1 Q=M=4096 D=128 K=32 split4 warp-select guard", "fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_largek_stress_qm4096_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}], ["fd9b_plus_1b34_k5", {"__dict_items__": [["candidate_id", "candidate_fd9b_plus_1b34_k5_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:candidate_k5_only_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:benchmark_candidate_k5_only_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge", "fcee_flashml_k5_bd4a_exact"]}], ["new_seed_ids", {"__tuple__": ["fcee_flashml_k5_bd4a_exact"]}], ["guard_plan", {"__tuple__": ["1b34/FCEE exact BF16 FlashML build B=1 Q=M=256 D=128 K=5 guard", "fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "flashml_correctness_b1_q256_m256_d128_k5"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}], ["fd9b_plus_01bb_2425_bd76_k20_9334_k32", {"__dict_items__": [["candidate_id", "candidate_fd9b_plus_01bb_2425_bd76_k20_9334_k32_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:candidate_floor_core_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:benchmark_candidate_floor_core_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge", "q4096_k8_lowfloor_fd9b_v3_exact_prefill_s4", "rag_microbatch_k10_q4_s144_g12_d555_v1", "rag_microbatch_k10_q8_s144_g12_d5ac_v1", "rag_microbatch_k10_q32_m64_s128_g8_4757_v1", "large_square_k20_efe4_bd76_split2_warp8", "overk_largek_q4096_k32_9334_v1"]}], ["new_seed_ids", {"__tuple__": ["q4096_k8_lowfloor_fd9b_v3_exact_prefill_s4", "rag_microbatch_k10_q4_s144_g12_d555_v1", "rag_microbatch_k10_q8_s144_g12_d5ac_v1", "rag_microbatch_k10_q32_m64_s128_g8_4757_v1", "large_square_k20_efe4_bd76_split2_warp8", "overk_largek_q4096_k32_9334_v1"]}], ["guard_plan", {"__tuple__": ["01bb exact BF16 build B=1 Q=M=4096 D=128 K=8 split4 exact-prefill guard", "2425 exact BF16 RAG microbatch K10 q4/q8/q32 guard", "bd76 exact BF16 build B=1 Q=M=8192 D=128 K=20 split2 warp8 guard", "9334 exact BF16 build B=1 Q=M=4096 D=128 K=32 split4 warp-select guard", "fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "build_qm4096_d128_k8", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_largek_stress_qm4096_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}], ["fd9b_plus_01bb_2425_1b34_k5_bd76_k20_9334_k32", {"__dict_items__": [["candidate_id", "candidate_fd9b_plus_01bb_2425_1b34_k5_bd76_k20_9334_k32_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:candidate_floor_core_k5_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:benchmark_candidate_floor_core_k5_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge", "q4096_k8_lowfloor_fd9b_v3_exact_prefill_s4", "rag_microbatch_k10_q4_s144_g12_d555_v1", "rag_microbatch_k10_q8_s144_g12_d5ac_v1", "rag_microbatch_k10_q32_m64_s128_g8_4757_v1", "large_square_k20_efe4_bd76_split2_warp8", "overk_largek_q4096_k32_9334_v1", "fcee_flashml_k5_bd4a_exact"]}], ["new_seed_ids", {"__tuple__": ["q4096_k8_lowfloor_fd9b_v3_exact_prefill_s4", "rag_microbatch_k10_q4_s144_g12_d555_v1", "rag_microbatch_k10_q8_s144_g12_d5ac_v1", "rag_microbatch_k10_q32_m64_s128_g8_4757_v1", "large_square_k20_efe4_bd76_split2_warp8", "overk_largek_q4096_k32_9334_v1", "fcee_flashml_k5_bd4a_exact"]}], ["guard_plan", {"__tuple__": ["01bb exact BF16 build B=1 Q=M=4096 D=128 K=8 split4 exact-prefill guard", "2425 exact BF16 RAG microbatch K10 q4/q8/q32 guard", "1b34/FCEE exact BF16 FlashML build B=1 Q=M=256 D=128 K=5 guard", "bd76 exact BF16 build B=1 Q=M=8192 D=128 K=20 split2 warp8 guard", "9334 exact BF16 build B=1 Q=M=4096 D=128 K=32 split4 warp-select guard", "fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "build_qm4096_d128_k8", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_largek_stress_qm4096_k32", "flashml_correctness_b1_q256_m256_d128_k5"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]]}'))
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "candidate_1877_plus_9a17_fp16_fd37_lowfloor_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:benchmark_candidate_fp16_fd37_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge"]}], ["guard_plan", {"__tuple__": ["fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session fd9b full90 champion baseline"]]}, {"__dict_items__": [["id", "candidate_fd9b_plus_bd76_k20_9334_k32_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:benchmark_candidate_k20_k32_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge", "large_square_k20_efe4_bd76_split2_warp8", "overk_largek_q4096_k32_9334_v1"]}], ["guard_plan", {"__tuple__": ["bd76 exact BF16 build B=1 Q=M=8192 D=128 K=20 split2 warp8 guard", "9334 exact BF16 build B=1 Q=M=4096 D=128 K=32 split4 warp-select guard", "fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_largek_stress_qm4096_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_fd9b_plus_1b34_k5_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:benchmark_candidate_k5_only_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge", "fcee_flashml_k5_bd4a_exact"]}], ["guard_plan", {"__tuple__": ["1b34/FCEE exact BF16 FlashML build B=1 Q=M=256 D=128 K=5 guard", "fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "flashml_correctness_b1_q256_m256_d128_k5"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_fd9b_plus_01bb_2425_bd76_k20_9334_k32_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:benchmark_candidate_floor_core_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge", "q4096_k8_lowfloor_fd9b_v3_exact_prefill_s4", "rag_microbatch_k10_q4_s144_g12_d555_v1", "rag_microbatch_k10_q8_s144_g12_d5ac_v1", "rag_microbatch_k10_q32_m64_s128_g8_4757_v1", "large_square_k20_efe4_bd76_split2_warp8", "overk_largek_q4096_k32_9334_v1"]}], ["guard_plan", {"__tuple__": ["01bb exact BF16 build B=1 Q=M=4096 D=128 K=8 split4 exact-prefill guard", "2425 exact BF16 RAG microbatch K10 q4/q8/q32 guard", "bd76 exact BF16 build B=1 Q=M=8192 D=128 K=20 split2 warp8 guard", "9334 exact BF16 build B=1 Q=M=4096 D=128 K=32 split4 warp-select guard", "fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "build_qm4096_d128_k8", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_largek_stress_qm4096_k32"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}, {"__dict_items__": [["id", "candidate_fd9b_plus_01bb_2425_1b34_k5_bd76_k20_9334_k32_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_fd9b_floor_seed_portfolio_5720_full90_synthesis_v1:benchmark_candidate_floor_core_k5_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge", "q4096_k8_lowfloor_fd9b_v3_exact_prefill_s4", "rag_microbatch_k10_q4_s144_g12_d555_v1", "rag_microbatch_k10_q8_s144_g12_d5ac_v1", "rag_microbatch_k10_q32_m64_s128_g8_4757_v1", "large_square_k20_efe4_bd76_split2_warp8", "overk_largek_q4096_k32_9334_v1", "fcee_flashml_k5_bd4a_exact"]}], ["guard_plan", {"__tuple__": ["01bb exact BF16 build B=1 Q=M=4096 D=128 K=8 split4 exact-prefill guard", "2425 exact BF16 RAG microbatch K10 q4/q8/q32 guard", "1b34/FCEE exact BF16 FlashML build B=1 Q=M=256 D=128 K=5 guard", "bd76 exact BF16 build B=1 Q=M=8192 D=128 K=20 split2 warp8 guard", "9334 exact BF16 build B=1 Q=M=4096 D=128 K=32 split4 warp-select guard", "fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "build_qm4096_d128_k8", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_largek_stress_qm4096_k32", "flashml_correctness_b1_q256_m256_d128_k5"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["rejected_reason", null]]}]}'))

def route_for_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> str:
    _candidate_config(candidate_key)
    if force_fallback:
        return _base_route(inputs, force_fallback=True)
    if candidate_key == BASE_CANDIDATE_KEY:
        return _base_route(inputs)
    seed_id = _matched_seed(inputs, candidate_key)
    if seed_id is not None:
        return _seed_route(seed_id, inputs)
    return _base_route(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> None:
    _candidate_config(candidate_key)
    if force_fallback:
        _base_launch(inputs, force_fallback=True)
        return
    if candidate_key == BASE_CANDIDATE_KEY:
        _base_launch(inputs)
        return
    seed_id = _matched_seed(inputs, candidate_key)
    if seed_id is not None:
        _seed_launch(seed_id, inputs)
        return
    _base_launch(inputs)

def candidate_baseline_fd9b_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=BASE_CANDIDATE_KEY)

def candidate_k20_k32_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_K20_K32)

def candidate_k5_only_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_K5_ONLY)

def candidate_floor_core_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_FLOOR_CORE)

def candidate_floor_core_k5_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_FLOOR_CORE_K5)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_floor_core_k5_full90_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=DEFAULT_CANDIDATE_KEY, force_fallback=True)

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], None]:
    if candidate_key == BASE_CANDIDATE_KEY:
        return candidate_baseline_fd9b_full90_v1
    if candidate_key == CANDIDATE_K20_K32:
        return candidate_k20_k32_full90_v1
    if candidate_key == CANDIDATE_K5_ONLY:
        return candidate_k5_only_full90_v1
    if candidate_key == CANDIDATE_FLOOR_CORE:
        return candidate_floor_core_full90_v1
    if candidate_key == CANDIDATE_FLOOR_CORE_K5:
        return candidate_floor_core_k5_full90_v1
    raise ValueError(''.join(['unknown fd9b floor-seed portfolio candidate ', format(repr(candidate_key), '')]))

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=FLOOR_CORE_K5_TARGET_SHAPES, benchmark: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark, kernel_fn=_candidate_kernel_fn(candidate_key))
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return fd9b._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _seed_trace_record(inputs: dict[str, Any], seed_id: str, *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    base_row = _base_trace_row(label, force_fallback=False)
    guard_id, guard_condition = _seed_guard(seed_id)
    if force_fallback:
        row = _base_trace_row(label, force_fallback=True)
        row['expected_seed'] = seed_id
        row['guard_id'] = ''.join(['forced_fallback_', format(seed_id, ''), '_disabled'])
        row['guard_condition'] = ''.join(['forced fallback to fd9b; ', format(seed_id, ''), ' disabled'])
        row['classification'] = 'guard-miss'
        row['parent_dispatcher_route'] = base_row.get('selected_route')
        row['baseline_dispatcher_route'] = base_row.get('selected_route')
        row['targeted_seed_row'] = TARGETED_SEED_ROWS.get(seed_id, {})
        return _normalize_route_row(row)
    return _normalize_route_row({'shape_key': label, 'selected_route': _seed_route(seed_id, inputs), 'selected_entrypoint': _seed_entrypoint(seed_id), 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': guard_id, 'guard_condition': guard_condition, 'coverage': 'fd9b floor-seed portfolio overlay before fd9b fallback', 'consumed_seed': seed_id, 'replaced_route': base_row.get('selected_route'), 'parent_dispatcher_route': base_row.get('selected_route'), 'parent_dispatcher_selected_seed': base_row.get('selected_seed'), 'baseline_dispatcher_route': base_row.get('selected_route'), 'targeted_seed_row': TARGETED_SEED_ROWS.get(seed_id, {}), 'classification': 'unmeasured'})

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    _candidate_config(candidate_key)
    label = str(inputs.get('label'))
    if candidate_key == BASE_CANDIDATE_KEY:
        return _normalize_route_row(_base_trace_row(label, force_fallback=force_fallback))
    seed_id = _matched_seed(inputs, candidate_key)
    if seed_id is not None:
        return _seed_trace_record(inputs, seed_id, force_fallback=force_fallback)
    row = _base_trace_row(label, force_fallback=force_fallback)
    row['expected_seed'] = _expected_seed(inputs, candidate_key)
    row['baseline_dispatcher_route'] = _base_route(inputs, force_fallback=force_fallback)
    row['parent_dispatcher_route'] = _base_route(inputs, force_fallback=force_fallback)
    return _normalize_route_row(row)

def route_trace_for_contract_shapes(shape_labels=None, *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> list[dict[str, Any]]:
    _candidate_config(candidate_key)
    labels = _shape_labels(shape_labels)
    return [_route_trace_record(_inputs_for_label(label), candidate_key=candidate_key, force_fallback=force_fallback) for label in labels]

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str, speedup_floor: float) -> list[dict[str, Any]]:
    new_seed_ids = set(_candidate_config(candidate_key)['new_seed_ids'])
    annotated = []
    for row in route_trace:
        label = str(row['shape_key'])
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms')
        selected_new_seed = row.get('selected_seed') in new_seed_ids
        speedup_vs_baseline = baseline_ms / candidate_ms if candidate_ms and baseline_ms else None
        speedup_vs_external = flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None
        out = dict(row)
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_fd9b_kernel_ms'] = baseline_ms
        out['shape_specific_kernel_ms'] = candidate_ms if selected_new_seed else None
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['relative_speedup_vs_fd9b'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['timing_backend'] = candidate_row.get('timing_backend') or baseline_row.get('timing_backend')
        out['route_changed_vs_fd9b'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        if row.get('expected_seed') in new_seed_ids and row.get('selected_seed') != row.get('expected_seed'):
            out['classification'] = 'guard-miss'
        elif selected_new_seed:
            if speedup_vs_external is not None and speedup_vs_external < speedup_floor:
                out['classification'] = 'kernel-slow'
            elif speedup_vs_baseline is not None and speedup_vs_baseline < 1.0:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'seed-consumed'
        elif speedup_vs_external is not None and speedup_vs_external < speedup_floor:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        elif candidate_ms is None:
            out['classification'] = 'unmeasured'
        else:
            out['classification'] = 'route-ok'
        annotated.append(_normalize_route_row(out))
    return annotated

def _below_flashlib_rows(report: dict[str, Any], route_trace: list[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if not isinstance(ratio, (float, int)) or ratio >= floor:
            continue
        trace_row = trace_by_label.get(str(label), {})
        classification = trace_row.get('classification', 'unmeasured')
        if classification in ('route-ok', 'unmeasured') and (not trace_row.get('selected_seed')):
            classification = 'fallback-slow'
        elif classification in ('route-ok', 'unmeasured') and trace_row.get('selected_seed'):
            classification = 'kernel-slow'
        rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_seed': trace_row.get('selected_seed'), 'expected_seed': trace_row.get('expected_seed'), 'selected_route': trace_row.get('selected_route'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': classification})
    return rows

def _seed_delta_matrix(candidate_key: str, candidate_report: dict[str, Any], baseline_report: dict[str, Any], labels: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    audit_labels = tuple(dict.fromkeys((*labels, *K48_EXCLUDED_TARGET_SHAPES)))
    for label in audit_labels:
        inputs = _inputs_for_label(label)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        selected_seed = _expected_seed(inputs, candidate_key)
        consumption_status = 'seed-consumed' if selected_seed in _candidate_config(candidate_key)['new_seed_ids'] else 'not-targeted'
        if label in K48_EXCLUDED_TARGET_SHAPES:
            selected_seed = SEED_E087_K48_ID
            consumption_status = 'excluded_aggregate_regression'
        rows.append({'shape_key': label, 'baseline_route': _base_route(inputs), 'candidate_route': route_for_contract_inputs(inputs, candidate_key=candidate_key), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'baseline_fd9b_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms'), 'delta_ms_candidate_minus_fd9b': candidate_ms - baseline_ms if candidate_ms is not None and baseline_ms is not None else None, 'speedup_vs_fd9b': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib') or baseline_row.get('ratio_vs_flashlib'), 'targeted_seed_row': TARGETED_SEED_ROWS.get(selected_seed, {}), 'consumption_status': consumption_status, 'candidate_passed': candidate_row.get('passed'), 'baseline_passed': baseline_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return rows

def benchmark_baseline_fd9b_full90_v1(*, use_cupti: bool=True, shape_labels=None, time_flashlib: bool=True, benchmark_correctness: bool=True) -> dict[str, Any]:
    return _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_fd9b_full90_v1, time_flashlib=time_flashlib, correctness=benchmark_correctness)

def _baseline_payload(report: dict[str, Any], *, shape_labels, use_cupti: bool, time_flashlib: bool, benchmark_correctness: bool, speedup_floor: float) -> dict[str, Any]:
    labels = _shape_labels(shape_labels)
    route_trace = route_trace_for_contract_shapes(labels, candidate_key=BASE_CANDIDATE_KEY)
    below_1x = _below_flashlib_rows(report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(report, route_trace, floor=speedup_floor)
    return {'candidate_id': BASE_CANDIDATE_ID, 'candidate_key': BASE_CANDIDATE_KEY, 'measured_entrypoint': BASELINE_BENCHMARK_ENTRYPOINT, 'source_task': 'generalize-auto-tuning-knn-build-fd9b', 'tflops': report.get('summary', {}).get('primary_mean'), 'all_correct': report.get('summary', {}).get('all_correct'), 'performance_comparable': report.get('summary', {}).get('performance_comparable'), 'invalid_performance_reason': report.get('summary', {}).get('invalid_performance_reason'), 'timing_backend': _timing_backend_name(use_cupti), 'timing_backends': _timing_backends_for_reports(report), 'use_cupti': use_cupti, 'denominator': _denominator_name(shape_labels), 'measured_shape_labels': list(labels), 'route_entrypoint': BASE_ROUTE_ENTRYPOINT, 'selected_route_labels': tuple(BASE_CONFIG['expected_shape_wins']), 'route_trace': route_trace, 'route_trace_included': True, 'contract_summary': report.get('summary'), 'contract_performance': report.get('performance'), 'contract_correctness': report.get('correctness'), 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'baseline_payload': None, 'speedup_floor': speedup_floor, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'report': report}

def _benchmark_payload(candidate_key: str, candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, shape_labels, use_cupti: bool, time_flashlib: bool, benchmark_correctness: bool, speedup_floor: float) -> dict[str, Any]:
    labels = _shape_labels(shape_labels)
    candidate_metric = candidate_report.get('summary', {}).get('primary_mean')
    baseline_metric = baseline_report.get('summary', {}).get('primary_mean')
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(labels, candidate_key=candidate_key), candidate_report, baseline_report, candidate_key=candidate_key, speedup_floor=speedup_floor)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=speedup_floor)
    config = _candidate_config(candidate_key)
    return {'candidate_id': config['candidate_id'], 'candidate_key': candidate_key, 'baseline_candidate_id': BASE_CANDIDATE_ID, 'baseline_candidate_key': BASE_CANDIDATE_KEY, 'source_task': 'generalize-auto-tuning-knn-build-5720', 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta_vs_fd9b': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'all_correct': candidate_report.get('summary', {}).get('all_correct'), 'baseline_all_correct': baseline_report.get('summary', {}).get('all_correct'), 'performance_comparable': candidate_report.get('summary', {}).get('performance_comparable'), 'invalid_performance_reason': candidate_report.get('summary', {}).get('invalid_performance_reason'), 'measured_entrypoint': config['benchmark_entrypoint'], 'baseline_measured_entrypoint': BASELINE_BENCHMARK_ENTRYPOINT, 'route_entrypoint': ROUTE_ENTRYPOINT, 'selected_seeds': config['selected_seeds'], 'new_seed_ids': config['new_seed_ids'], 'selected_route_labels': config['expected_shape_wins'], 'selected_route_rows': _rows_for_labels(candidate_report, config['expected_shape_wins']), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, config['expected_shape_wins']), 'seed_delta_matrix': _seed_delta_matrix(candidate_key, candidate_report, baseline_report, labels), 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'rejected_route_combinations': REJECTED_ROUTE_COMBINATIONS, 'selected_candidate_dispatcher': config['candidate_id'], 'source_tasks': SOURCE_TASKS, 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, candidate_key=candidate_key, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report.get('summary'), 'baseline_contract_summary': baseline_report.get('summary'), 'contract_performance': candidate_report.get('performance'), 'baseline_contract_performance': baseline_report.get('performance'), 'contract_correctness': candidate_report.get('correctness'), 'timing_backend': _timing_backend_name(use_cupti), 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'denominator': _denominator_name(shape_labels), 'measured_shape_labels': list(labels), 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'baseline_payload': None, 'speedup_floor': speedup_floor, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'delta_vs_fd9b': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'valid_measurement_count': candidate_report.get('performance', {}).get('valid_measurement_count'), 'comparable': candidate_report.get('performance', {}).get('comparable')}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_portfolio(candidate_key: str=DEFAULT_CANDIDATE_KEY, *, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, time_flashlib: bool=True, benchmark_correctness: bool=True, speedup_floor: float=1.2) -> dict[str, Any]:
    if candidate_key == BASE_CANDIDATE_KEY:
        baseline = benchmark_baseline_fd9b_full90_v1(use_cupti=use_cupti, shape_labels=shape_labels, time_flashlib=time_flashlib, benchmark_correctness=benchmark_correctness)
        return _baseline_payload(baseline, shape_labels=shape_labels, use_cupti=use_cupti, time_flashlib=time_flashlib, benchmark_correctness=benchmark_correctness, speedup_floor=speedup_floor)
    baseline = baseline_report
    if baseline is None:
        baseline = benchmark_baseline_fd9b_full90_v1(use_cupti=use_cupti, shape_labels=shape_labels, time_flashlib=time_flashlib, benchmark_correctness=benchmark_correctness)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), time_flashlib=time_flashlib, correctness=benchmark_correctness)
    return _benchmark_payload(candidate_key, candidate_report, baseline, shape_labels=shape_labels, use_cupti=use_cupti, time_flashlib=time_flashlib, benchmark_correctness=benchmark_correctness, speedup_floor=speedup_floor)

def benchmark_candidate_k20_k32_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_K20_K32, **kwargs)

def benchmark_candidate_k5_only_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_K5_ONLY, **kwargs)

def benchmark_candidate_floor_core_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_FLOOR_CORE, **kwargs)

def benchmark_candidate_floor_core_k5_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_FLOOR_CORE_K5, **kwargs)

def _candidate_comparable(payload: dict[str, Any]) -> bool:
    return bool(payload.get('all_correct')) and bool(payload.get('performance_comparable'))

def _selected_candidate_key(payloads: dict[str, dict[str, Any]]) -> str | None:
    candidates = [key for key in payloads if key != BASE_CANDIDATE_KEY and _candidate_comparable(payloads[key])]
    if not candidates:
        return None
    return max(candidates, key=lambda key: (payloads[key].get('tflops') or float('-inf'), len(CANDIDATE_CONFIGS[key]['new_seed_ids'])))

def _summary_payload(*, payloads: dict[str, dict[str, Any]], artifacts: dict[str, str], shape_labels, time_flashlib: bool, benchmark_correctness: bool, speedup_floor: float) -> dict[str, Any]:
    selected_key = _selected_candidate_key(payloads)
    selected_payload = payloads.get(selected_key, {}) if selected_key else {}
    baseline_payload = payloads[BASE_CANDIDATE_KEY]
    baseline_value = baseline_payload.get('tflops')
    selected_value = selected_payload.get('tflops')
    metric_delta = selected_payload.get('metric_delta_vs_fd9b')
    below_floor = selected_payload.get('flashlib_parity_ledger', {}).get('rows_below_floor', [])
    no_regression = selected_key is not None and selected_value is not None and (baseline_value is not None) and (float(selected_value) >= float(baseline_value))
    return {'candidate_id': 'dispatcher_synthesis_fd9b_floor_seed_portfolio_5720_full90_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':write_benchmark_artifacts']), 'workflow_mode': 'generalize_auto_tuning', 'auto_tuning_stage': 'dispatcher-synthesis', 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'baseline_candidate_key': BASE_CANDIDATE_KEY, 'selected_candidate_key': selected_key, 'selected_candidate_dispatcher': _candidate_id(selected_key) if selected_key else None, 'default_candidate_key': DEFAULT_CANDIDATE_KEY, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'rejected_route_combinations': REJECTED_ROUTE_COMBINATIONS, 'candidate_rankings': [{'candidate_key': key, 'candidate_id': payloads[key].get('candidate_id'), 'tflops': payloads[key].get('tflops'), 'delta_vs_fd9b': payloads[key].get('metric_delta_vs_fd9b', 0.0), 'all_correct': payloads[key].get('all_correct'), 'performance_comparable': payloads[key].get('performance_comparable'), 'performance_coverage': payloads[key].get('performance_coverage')} for key in CANDIDATE_KEYS if key in payloads], 'seed_delta_matrix': selected_payload.get('seed_delta_matrix', []), 'seed_delta_matrix_all_candidates': {key: payloads[key].get('seed_delta_matrix', []) for key in payloads if key != BASE_CANDIDATE_KEY}, 'full_denominator_ab': {'baseline_payload': artifacts['baseline_payload'], 'candidate_payload': artifacts.get(''.join([format(selected_key, ''), '_payload'])) if selected_key else None, 'denominator': _denominator_name(shape_labels), 'timing_backend': selected_payload.get('timing_backend', baseline_payload.get('timing_backend')), 'metric_delta': metric_delta, 'route_trace': selected_payload.get('route_trace', [])}, 'promotion_gate_summary': {'full_dispatch_denominator': bool(selected_payload.get('all_correct')), 'same_session_no_regression': no_regression, 'performance_coverage': 'pass' if selected_key and (not below_floor) else 'partial', 'hot_bucket_parity': bool(selected_key) and (not below_floor), 'expanded_shape_coverage': False}, 'flashlib_parity_ledger': selected_payload.get('flashlib_parity_ledger', baseline_payload.get('flashlib_parity_ledger', {})), 'benchmark_payload_artifacts': artifacts, 'route_trace': selected_payload.get('route_trace', []), 'baseline_tflops': baseline_value, 'selected_tflops': selected_value, 'metric_delta_vs_fd9b': metric_delta, 'selected_no_regression': no_regression}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, time_flashlib: bool=True, benchmark_correctness: bool=True, candidate_keys: tuple[str, ...] | None=None, speedup_floor: float=1.2) -> dict[str, str]:
    labels = _shape_labels(shape_labels)
    denom_label = _denominator_label(shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_report = benchmark_baseline_fd9b_full90_v1(use_cupti=use_cupti, shape_labels=shape_labels, time_flashlib=time_flashlib, benchmark_correctness=benchmark_correctness)
    baseline_payload = _baseline_payload(baseline_report, shape_labels=shape_labels, use_cupti=use_cupti, time_flashlib=time_flashlib, benchmark_correctness=benchmark_correctness, speedup_floor=speedup_floor)
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_fd9b_v1.json'])
    baseline_payload['flashlib_parity_ledger']['baseline_payload'] = str(baseline_path)
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts: dict[str, str] = {'baseline_payload': str(baseline_path)}
    payloads = {BASE_CANDIDATE_KEY: baseline_payload}
    selected_candidate_keys = list(DEFAULT_SYNTHESIS_CANDIDATES) if candidate_keys is None else list(candidate_keys)
    for candidate_key in selected_candidate_keys:
        if candidate_key == BASE_CANDIDATE_KEY:
            raise ValueError('candidate_keys must not include the baseline key')
        _candidate_config(candidate_key)
        candidate_payload = benchmark_candidate_portfolio(candidate_key, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, time_flashlib=time_flashlib, benchmark_correctness=benchmark_correctness, speedup_floor=speedup_floor)
        payloads[candidate_key] = candidate_payload
        candidate_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_', format(candidate_key, ''), '_v1.json'])
        trace_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_', format(candidate_key, ''), '_v1.json'])
        forced_trace_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_', format(candidate_key, ''), '_v1.json'])
        seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_', format(candidate_key, ''), '_v1.json'])
        candidate_payload['flashlib_parity_ledger']['baseline_payload'] = str(baseline_path)
        candidate_path.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        trace_path.write_text(json.dumps(candidate_payload['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
        forced_trace_path.write_text(json.dumps(candidate_payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
        seed_matrix_path.write_text(json.dumps(candidate_payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
        artifacts[''.join([format(candidate_key, ''), '_payload'])] = str(candidate_path)
        artifacts[''.join([format(candidate_key, ''), '_route_trace'])] = str(trace_path)
        artifacts[''.join([format(candidate_key, ''), '_forced_fallback_trace'])] = str(forced_trace_path)
        artifacts[''.join([format(candidate_key, ''), '_seed_delta_matrix'])] = str(seed_matrix_path)
    summary = _summary_payload(payloads=payloads, artifacts=artifacts, shape_labels=shape_labels, time_flashlib=time_flashlib, benchmark_correctness=benchmark_correctness, speedup_floor=speedup_floor)
    summary_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_synthesis_fd9b_floor_seed_portfolio_5720_v1.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['dispatcher_synthesis'] = str(summary_path)
    artifacts['measured_shape_count'] = str(len(labels))
    return artifacts
