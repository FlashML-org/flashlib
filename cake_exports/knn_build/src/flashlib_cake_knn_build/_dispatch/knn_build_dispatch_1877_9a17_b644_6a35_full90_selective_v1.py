"""Full90 selective synthesis over 1877 with 9a17, b644, and 6a35 seeds.

Minimum target architecture: sm_100a. This additive dispatcher wrapper keeps
the variance-audited 1877 full90 sidecar as the baseline route, then measures
guarded portfolios that consume the 9a17 residual RAG/search seed, the b644
exact-five build low-floor seed, and the 6a35 D64/Q4096 exact seed.

Production routes stay Weave-only; FlashLib is timed only by the contract
harness as a black-box reference.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_build_lowfloor_d43e_v1 as build_b644
from . import knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1 as base1877
from . import knn_build_d64_q4096_c271_twostage_v1 as d64_6a35
from . import knn_build_residual_rag_search_1877_v1 as rag_9a17
MODULE = 'loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1'
eval_mod = base1877.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
BASE_1877_KEY = base1877.DEFAULT_CANDIDATE_KEY
BASE_1877_CONFIG = base1877.CANDIDATE_CONFIGS[BASE_1877_KEY]
BASE_1877_ID = BASE_1877_CONFIG['candidate_id']
BASE_1877_ENTRYPOINT = BASE_1877_CONFIG['benchmark_entrypoint']
BASE_1877_ROUTE_ENTRYPOINT = base1877.ROUTE_ENTRYPOINT
CANDIDATE_9A17_ONLY = '1877_plus_9a17_residual_rag_search'
CANDIDATE_B644_ONLY = '1877_plus_b644_exact5'
CANDIDATE_6A35_D64_ONLY = '1877_plus_6a35_d64'
CANDIDATE_B644_6A35 = '1877_plus_b644_exact5_6a35_d64'
CANDIDATE_9A17_B644 = '1877_plus_9a17_b644_exact5'
CANDIDATE_9A17_6A35 = '1877_plus_9a17_6a35_d64'
CANDIDATE_9A17_B644_6A35 = '1877_plus_9a17_b644_exact5_6a35_d64'
DEFAULT_CANDIDATE_KEY = CANDIDATE_9A17_B644_6A35
CANDIDATE_KEYS = (BASE_1877_KEY, CANDIDATE_9A17_ONLY, CANDIDATE_B644_ONLY, CANDIDATE_6A35_D64_ONLY, CANDIDATE_B644_6A35, CANDIDATE_9A17_B644, CANDIDATE_9A17_6A35, CANDIDATE_9A17_B644_6A35)
DEFAULT_SYNTHESIS_CANDIDATES = (CANDIDATE_9A17_ONLY, CANDIDATE_B644_ONLY, CANDIDATE_6A35_D64_ONLY, CANDIDATE_B644_6A35, CANDIDATE_9A17_B644_6A35)
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
CANDIDATE_9A17_ONLY_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_9a17_only_full90_v1'])
CANDIDATE_B644_ONLY_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_b644_only_full90_v1'])
CANDIDATE_6A35_D64_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_6a35_d64_only_full90_v1'])
CANDIDATE_B644_6A35_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_b644_6a35_full90_v1'])
CANDIDATE_9A17_B644_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_9a17_b644_full90_v1'])
CANDIDATE_9A17_6A35_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_9a17_6a35_full90_v1'])
CANDIDATE_COMBINED_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_9a17_b644_6a35_full90_v1'])
BASE_1877_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10"]}'))
RAG_9A17_TARGET_SHAPES = rag_9a17.TARGET_SHAPES
B644_TARGET_SHAPES = build_b644.TARGET_SHAPES
D64_6A35_TARGET_SHAPES = d64_6a35.TARGET_SHAPES
RAG_ONLY_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10"]}'))
B644_ONLY_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20"]}'))
D64_ONLY_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}'))
B644_D64_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}'))
RAG_B644_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20"]}'))
RAG_D64_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}'))
COMBINED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}'))
SEED_9A17_Q128_ID = rag_9a17.SEED_Q128_ID
SEED_9A17_RECT_ID = rag_9a17.SEED_RECT_ID
SEED_B644_Q512_K2_ID = build_b644.SEED_Q512_K2_ID
SEED_B644_Q4096_K8_ID = build_b644.SEED_Q4096_K8_ID
SEED_B644_MIDK_ID = build_b644.SEED_MIDK_ID
SEED_B644_TAIL_K20_ID = build_b644.SEED_TAIL_K20_ID
SEED_6A35_D64_ID = 'd64_q4096_c271_split4_unordered_sortedemit_v1'
SOURCE_TASKS = {**base1877.SOURCE_TASKS, **rag_9a17.SOURCE_TASKS, **build_b644.SOURCE_TASKS, SEED_6A35_D64_ID: 'weave-evolve-knn-build-6a35 / design_doc/active/weave_evolve_knn_build_round_143_weave-evolve-knn-build-c271-d64q4096-prodaxis.md'}
PRODUCTION_ROUTE_MODULES = {**base1877.PRODUCTION_ROUTE_MODULES, **rag_9a17.PRODUCTION_ROUTE_MODULES, **build_b644.PRODUCTION_ROUTE_MODULES, SEED_6A35_D64_ID: ''.join([format(d64_6a35.MODULE, ''), ':launch_from_contract_inputs']), BASE_1877_ID: BASE_1877_ROUTE_ENTRYPOINT}
TARGETED_SEED_ROWS = {**base1877.TARGETED_SEED_ROWS, SEED_9A17_Q128_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_residual_rag_search_1877_v1/residual_rag_search_1877_v1.json', 'shape_labels': (rag_9a17.RAG_Q128_K32,), 'source_task': 'weave-evolve-knn-build-9a17'}, SEED_9A17_RECT_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_residual_rag_search_1877_v1/residual_rag_search_1877_v1.json', 'shape_labels': (rag_9a17.SEARCH_RECT_Q1024_K10,), 'source_task': 'weave-evolve-knn-build-9a17'}, SEED_B644_Q512_K2_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_build_lowfloor_d43e_v1/build_lowfloor_d43e_v1.json', 'shape_labels': (build_b644.TARGET_Q512_K2,), 'source_task': 'weave-evolve-knn-build-b644'}, SEED_B644_Q4096_K8_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_build_lowfloor_d43e_v1/build_lowfloor_d43e_v1.json', 'shape_labels': (build_b644.TARGET_Q4096_K8,), 'source_task': 'weave-evolve-knn-build-b644'}, SEED_B644_MIDK_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_build_lowfloor_d43e_v1/build_lowfloor_d43e_v1.json', 'shape_labels': build_b644.TARGET_MIDK_SHAPES, 'source_task': 'weave-evolve-knn-build-b644'}, SEED_B644_TAIL_K20_ID: {'source_payload': 'artifacts/weave_evolve/knn_build_build_lowfloor_d43e_v1/build_lowfloor_d43e_v1.json', 'shape_labels': (build_b644.TARGET_TAIL_Q3072_K20,), 'source_task': 'weave-evolve-knn-build-b644'}, SEED_6A35_D64_ID: {'source_payload': 'design_doc/active/weave_evolve_knn_build_round_143_weave-evolve-knn-build-c271-d64q4096-prodaxis.md', 'shape_labels': D64_6A35_TARGET_SHAPES, 'source_task': 'weave-evolve-knn-build-6a35'}}
REJECTED_ROUTE_COMBINATIONS = (*base1877.REJECTED_ROUTE_COMBINATIONS, {'id': 'b3ec_dd3e_combined_cca8_not_replayed', 'entrypoint': 'loom.examples.weave.knn_build_dispatch_1877_9a17_b3ec_dd3e_full90_synthesis_v1', 'status': 'negative_read_ref', 'source_task': 'generalize-auto-tuning-knn-build-cca8', 'reason': 'cca8 measured b3ec/dd3e additive portfolios as full90 regressions; this lane replays b644/6a35 instead.'}, {'id': 'exact_five_duplicates_110d_11fc_5886', 'entrypoint': 'weave-evolve duplicate exact-five wrappers', 'status': 'dominated_read_ref', 'source_task': 'kernel-rank-generalize-auto-tuning-cohort-knn-build-fa86-0bbf', 'reason': '110d, 11fc, and 5886 duplicate the exact-five bucket and are superseded by b644.'})

def _select_contract_shapes(shape_labels):
    return base1877._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base1877._trace_inputs_for_shape(shape)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return base1877._normalize_route_row(row)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return _trace_inputs_for_shape(_select_contract_shapes((label,))[0])

def _timing_backend_name(use_cupti: bool) -> str:
    return base1877._timing_backend_name(use_cupti)

def _payload_shape_labels(shape_labels) -> str | tuple[str, ...]:
    return base1877._payload_shape_labels(shape_labels)

def _denominator_name(shape_labels) -> str:
    return base1877._denominator_name(shape_labels)

def _denominator_label(shape_labels) -> str:
    return base1877._denominator_label(shape_labels)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base1877._rows_for_labels(report, labels)

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base1877._timing_backends_for_reports(*reports)

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown full90 1877/9a17/b644/6a35 candidate ', format(repr(candidate_key), '')])) from exc

def _candidate_id(candidate_key: str | None) -> str | None:
    if candidate_key is None:
        return None
    return str(_candidate_config(candidate_key)['candidate_id'])

def _candidate_has_9a17(candidate_key: str) -> bool:
    return candidate_key in (CANDIDATE_9A17_ONLY, CANDIDATE_9A17_B644, CANDIDATE_9A17_6A35, CANDIDATE_9A17_B644_6A35)

def _candidate_has_b644(candidate_key: str) -> bool:
    return candidate_key in (CANDIDATE_B644_ONLY, CANDIDATE_B644_6A35, CANDIDATE_9A17_B644, CANDIDATE_9A17_B644_6A35)

def _candidate_has_6a35(candidate_key: str) -> bool:
    return candidate_key in (CANDIDATE_6A35_D64_ONLY, CANDIDATE_B644_6A35, CANDIDATE_9A17_6A35, CANDIDATE_9A17_B644_6A35)

def _matched_new_seed(inputs: dict[str, Any], candidate_key: str):
    if candidate_key == BASE_1877_KEY:
        return None
    if _candidate_has_9a17(candidate_key) and rag_9a17._selected_seed_for_inputs(inputs)[0] is not None:
        return rag_9a17
    if _candidate_has_b644(candidate_key) and build_b644._selected_seed_for_inputs(inputs)[0] is not None:
        return build_b644
    if _candidate_has_6a35(candidate_key) and d64_6a35._eligible_exact_q4096_d64(inputs):
        return d64_6a35
    return None

def _seed_expected_id(seed_module, inputs: dict[str, Any]) -> str | None:
    if seed_module is rag_9a17:
        return rag_9a17._selected_seed_for_inputs(inputs)[0]
    if seed_module is build_b644:
        return build_b644._selected_seed_for_inputs(inputs)[0]
    if seed_module is d64_6a35 and d64_6a35._eligible_exact_q4096_d64(inputs):
        return SEED_6A35_D64_ID
    return None

def _seed_route_for_module(seed_module, inputs: dict[str, Any]) -> str:
    if seed_module is d64_6a35:
        return d64_6a35.route_name_for_inputs(inputs)
    return seed_module.route_for_contract_inputs(inputs)

def _seed_launch_for_module(seed_module, inputs: dict[str, Any]) -> None:
    seed_module.launch_from_contract_inputs(inputs)

def _seed_trace_for_module(seed_module, label: str) -> dict[str, Any]:
    if seed_module is d64_6a35:
        inputs = _inputs_for_label(label)
        route = d64_6a35.route_name_for_inputs(inputs)
        return {'shape_key': label, 'selected_route': route, 'selected_entrypoint': ''.join([format(d64_6a35.MODULE, ''), ':launch_from_contract_inputs']), 'selected_seed': SEED_6A35_D64_ID, 'expected_seed': SEED_6A35_D64_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['selective_6a35_', format(SEED_6A35_D64_ID, '')]), 'guard_condition': 'exact BF16 build B=1 Q=M=4096 D=64 K=10 split4 unordered', 'matched_label': d64_6a35.TARGET_SHAPE, 'classification': 'seed-consumed'}
    return dict(seed_module.route_trace_for_contract_shapes((label,))[0])

def _base_route(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return base1877.route_for_contract_inputs(inputs, candidate_key=BASE_1877_KEY, force_fallback=force_fallback)

def _base_launch(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    base1877.launch_from_contract_inputs(inputs, candidate_key=BASE_1877_KEY, force_fallback=force_fallback)

def _base_trace_row(label: str, *, force_fallback: bool=False) -> dict[str, Any]:
    return dict(base1877.route_trace_for_contract_shapes((label,), candidate_key=BASE_1877_KEY, force_fallback=force_fallback)[0])

def _expected_seed(inputs: dict[str, Any], candidate_key: str) -> str | None:
    seed_module = _matched_new_seed(inputs, candidate_key)
    if seed_module is not None:
        return _seed_expected_id(seed_module, inputs)
    return base1877._expected_seed(inputs, BASE_1877_KEY)

def route_for_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> str:
    _candidate_config(candidate_key)
    if force_fallback:
        return _base_route(inputs, force_fallback=True)
    if candidate_key == BASE_1877_KEY:
        return _base_route(inputs)
    seed_module = _matched_new_seed(inputs, candidate_key)
    if seed_module is not None:
        return _seed_route_for_module(seed_module, inputs)
    return _base_route(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> None:
    _candidate_config(candidate_key)
    if force_fallback:
        _base_launch(inputs, force_fallback=True)
        return
    if candidate_key == BASE_1877_KEY:
        _base_launch(inputs)
        return
    seed_module = _matched_new_seed(inputs, candidate_key)
    if seed_module is not None:
        _seed_launch_for_module(seed_module, inputs)
        return
    _base_launch(inputs)

def candidate_parent_1877_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=BASE_1877_KEY)

def candidate_9a17_only_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_9A17_ONLY)

def candidate_b644_only_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_B644_ONLY)

def candidate_6a35_d64_only_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_6A35_D64_ONLY)

def candidate_b644_6a35_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_B644_6A35)

def candidate_9a17_b644_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_9A17_B644)

def candidate_9a17_6a35_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_9A17_6A35)

def candidate_9a17_b644_6a35_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_9A17_B644_6A35)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_9a17_b644_6a35_full90_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=DEFAULT_CANDIDATE_KEY, force_fallback=True)

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], None]:
    if candidate_key == BASE_1877_KEY:
        return candidate_parent_1877_full90_v1
    if candidate_key == CANDIDATE_9A17_ONLY:
        return candidate_9a17_only_full90_v1
    if candidate_key == CANDIDATE_B644_ONLY:
        return candidate_b644_only_full90_v1
    if candidate_key == CANDIDATE_6A35_D64_ONLY:
        return candidate_6a35_d64_only_full90_v1
    if candidate_key == CANDIDATE_B644_6A35:
        return candidate_b644_6a35_full90_v1
    if candidate_key == CANDIDATE_9A17_B644:
        return candidate_9a17_b644_full90_v1
    if candidate_key == CANDIDATE_9A17_6A35:
        return candidate_9a17_6a35_full90_v1
    if candidate_key == CANDIDATE_9A17_B644_6A35:
        return candidate_9a17_b644_6a35_full90_v1
    raise ValueError(''.join(['unknown full90 1877/9a17/b644/6a35 candidate ', format(repr(candidate_key), '')]))

def _selected_seeds(*seed_groups: tuple[str, ...]) -> tuple[str, ...]:
    values: list[str] = []
    for group in seed_groups:
        values.extend(group)
    return tuple(dict.fromkeys(values))
PARENT_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1"]}'))
RAG_9A17_SEEDS = (SEED_9A17_Q128_ID, SEED_9A17_RECT_ID)
B644_SEEDS = (SEED_B644_Q512_K2_ID, SEED_B644_Q4096_K8_ID, SEED_B644_MIDK_ID, SEED_B644_TAIL_K20_ID)
D64_6A35_SEEDS = (SEED_6A35_D64_ID,)
CANDIDATE_CONFIGS = _decode_capture(_json_loads('{"__dict_items__": [["ad64_plus_best_per_shape_build_k10_plus_ceb3", {"__dict_items__": [["candidate_id", "candidate_ad64_plus_best_build_k10_plus_ceb3_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_best_build_ceb3_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1"]}], ["new_seed_ids", {"__tuple__": []}], ["guard_plan", {"__tuple__": ["4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session 1877 full90 champion baseline"]]}], ["1877_plus_9a17_residual_rag_search", {"__dict_items__": [["candidate_id", "candidate_1877_plus_9a17_residual_rag_search_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_9a17_only_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_9a17_only_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1"]}], ["new_seed_ids", {"__tuple__": ["rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1"]}], ["guard_plan", {"__tuple__": ["9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "diagnostic replay of the cca8 9a17-only candidate"]]}], ["1877_plus_b644_exact5", {"__dict_items__": [["candidate_id", "candidate_1877_plus_b644_exact5_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_b644_only_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_b644_only_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4"]}], ["new_seed_ids", {"__tuple__": ["d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4"]}], ["guard_plan", {"__tuple__": ["b644 exact BF16 build B=1 Q=M=512 D=128 K=2 split4 guard", "b644 exact BF16 build B=1 Q=M=4096 D=128 K=8 split4 guard", "b644 exact BF16 build B=1 Q=M=2048 D=128 K in {11,13} guard", "b644 exact BF16 build B=1 Q=M=3072 D=128 K=20 split4 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "single seed-bank diagnostic; misses 9a17 and 6a35 D64 rows"]]}], ["1877_plus_6a35_d64", {"__dict_items__": [["candidate_id", "candidate_1877_plus_6a35_d64_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_6a35_d64_only_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_6a35_d64_only_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["new_seed_ids", {"__tuple__": ["d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["guard_plan", {"__tuple__": ["6a35 exact BF16 build B=1 Q=M=4096 D=64 K=10 split4 unordered guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "single exact D64 diagnostic; misses 9a17 and b644 rows"]]}], ["1877_plus_b644_exact5_6a35_d64", {"__dict_items__": [["candidate_id", "candidate_1877_plus_b644_exact5_6a35_d64_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_b644_6a35_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_b644_6a35_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4", "d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["new_seed_ids", {"__tuple__": ["d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4", "d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["guard_plan", {"__tuple__": ["b644 exact-five build low-floor guards", "6a35 exact BF16 build B=1 Q=M=4096 D=64 K=10 split4 unordered guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "build-only portfolio; selected only if full90 aggregate wins without 9a17"]]}], ["1877_plus_9a17_b644_exact5", {"__dict_items__": [["candidate_id", "candidate_1877_plus_9a17_b644_exact5_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_9a17_b644_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_9a17_b644_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4"]}], ["new_seed_ids", {"__tuple__": ["rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4"]}], ["guard_plan", {"__tuple__": ["9a17 exact-two residual RAG/search guards", "b644 exact-five build low-floor guards", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "legal combination; measured only when requested"]]}], ["1877_plus_9a17_6a35_d64", {"__dict_items__": [["candidate_id", "candidate_1877_plus_9a17_6a35_d64_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_9a17_6a35_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_9a17_6a35_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["new_seed_ids", {"__tuple__": ["rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["guard_plan", {"__tuple__": ["9a17 exact-two residual RAG/search guards", "6a35 exact BF16 build B=1 Q=M=4096 D=64 K=10 split4 unordered guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "legal combination; measured only when requested"]]}], ["1877_plus_9a17_b644_exact5_6a35_d64", {"__dict_items__": [["candidate_id", "candidate_1877_plus_9a17_b644_exact5_6a35_d64_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_9a17_b644_6a35_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_9a17_b644_6a35_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4", "d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["new_seed_ids", {"__tuple__": ["rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4", "d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["guard_plan", {"__tuple__": ["9a17 exact-two residual RAG/search guards", "b644 exact-five build low-floor guards", "6a35 exact BF16 build B=1 Q=M=4096 D=64 K=10 split4 unordered guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "selected only if same-session full90 no-regression passes"]]}]]}'))
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "candidate_ad64_plus_best_build_k10_plus_ceb3_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:benchmark_candidate_best_build_ceb3_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1"]}], ["guard_plan", {"__tuple__": ["4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "same-session 1877 full90 champion baseline"]]}, {"__dict_items__": [["id", "candidate_1877_plus_9a17_residual_rag_search_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_9a17_only_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1"]}], ["guard_plan", {"__tuple__": ["9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "diagnostic replay of the cca8 9a17-only candidate"]]}, {"__dict_items__": [["id", "candidate_1877_plus_b644_exact5_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_b644_only_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4"]}], ["guard_plan", {"__tuple__": ["b644 exact BF16 build B=1 Q=M=512 D=128 K=2 split4 guard", "b644 exact BF16 build B=1 Q=M=4096 D=128 K=8 split4 guard", "b644 exact BF16 build B=1 Q=M=2048 D=128 K in {11,13} guard", "b644 exact BF16 build B=1 Q=M=3072 D=128 K=20 split4 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "single seed-bank diagnostic; misses 9a17 and 6a35 D64 rows"]]}, {"__dict_items__": [["id", "candidate_1877_plus_6a35_d64_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_6a35_d64_only_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["guard_plan", {"__tuple__": ["6a35 exact BF16 build B=1 Q=M=4096 D=64 K=10 split4 unordered guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "single exact D64 diagnostic; misses 9a17 and b644 rows"]]}, {"__dict_items__": [["id", "candidate_1877_plus_b644_exact5_6a35_d64_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_b644_6a35_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4", "d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["guard_plan", {"__tuple__": ["b644 exact-five build low-floor guards", "6a35 exact BF16 build B=1 Q=M=4096 D=64 K=10 split4 unordered guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "build-only portfolio; selected only if full90 aggregate wins without 9a17"]]}, {"__dict_items__": [["id", "candidate_1877_plus_9a17_b644_exact5_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_9a17_b644_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4"]}], ["guard_plan", {"__tuple__": ["9a17 exact-two residual RAG/search guards", "b644 exact-five build low-floor guards", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "legal combination; measured only when requested"]]}, {"__dict_items__": [["id", "candidate_1877_plus_9a17_6a35_d64_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_9a17_6a35_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["guard_plan", {"__tuple__": ["9a17 exact-two residual RAG/search guards", "6a35 exact BF16 build B=1 Q=M=4096 D=64 K=10 split4 unordered guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "legal combination; measured only when requested"]]}, {"__dict_items__": [["id", "candidate_1877_plus_9a17_b644_exact5_6a35_d64_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_9a17_b644_6a35_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "d43e_lowk_q512_k2_s4", "c3bf_direct_v20_q4096_k8_s4", "d43e_midk_k11k13_e080", "b3ec_v20_q3072_k20_s4", "d64_q4096_c271_split4_unordered_sortedemit_v1"]}], ["guard_plan", {"__tuple__": ["9a17 exact-two residual RAG/search guards", "b644 exact-five build low-floor guards", "6a35 exact BF16 build B=1 Q=M=4096 D=64 K=10 split4 unordered guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_k_sweep_qm512_k2", "build_qm4096_d128_k8", "build_k_sweep_qm2048_k11", "build_k_sweep_qm2048_k13", "build_tail_b1_q3072_m3072_d128_k20", "build_dim_sweep_b1_q4096_m4096_d64_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_ad64_1b8f_4b51_ceb3_full90_synthesis_v1:launch_from_contract_inputs"], ["rejected_reason", "selected only if same-session full90 no-regression passes"]]}]}'))

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=COMBINED_TARGET_SHAPES, benchmark: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark, kernel_fn=_candidate_kernel_fn(candidate_key))
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return base1877._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _seed_trace_record(inputs: dict[str, Any], *, seed_module, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    seed_id = _seed_expected_id(seed_module, inputs)
    base_row = _base_trace_row(label, force_fallback=False)
    if force_fallback:
        row = _base_trace_row(label, force_fallback=True)
        row['expected_seed'] = seed_id
        row['guard_id'] = ''.join(['forced_fallback_', format(seed_id, ''), '_disabled'])
        row['guard_condition'] = ''.join(['forced fallback to 1877 baseline; ', format(seed_id, ''), ' disabled'])
        row['classification'] = 'guard-miss'
        row['parent_dispatcher_route'] = base_row.get('selected_route')
        row['baseline_dispatcher_route'] = base_row.get('selected_route')
        return _normalize_route_row(row)
    row = _seed_trace_for_module(seed_module, label)
    row['expected_seed'] = seed_id
    row['parent_dispatcher_route'] = base_row.get('selected_route')
    row['parent_dispatcher_selected_seed'] = base_row.get('selected_seed')
    row['baseline_dispatcher_route'] = base_row.get('selected_route')
    row['targeted_seed_row'] = TARGETED_SEED_ROWS.get(seed_id, {})
    row['coverage'] = 'full90 selective seed overlay before 1877 full90 baseline'
    row['classification'] = 'unmeasured'
    return _normalize_route_row(row)

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    if candidate_key == BASE_1877_KEY:
        return _normalize_route_row(_base_trace_row(label, force_fallback=force_fallback))
    seed_module = _matched_new_seed(inputs, candidate_key)
    if seed_module is not None:
        return _seed_trace_record(inputs, seed_module=seed_module, force_fallback=force_fallback)
    row = _base_trace_row(label, force_fallback=force_fallback)
    row['parent_dispatcher_route'] = _base_route(inputs, force_fallback=force_fallback)
    row['expected_seed'] = _expected_seed(inputs, candidate_key)
    return _normalize_route_row(row)

def route_trace_for_contract_shapes(shape_labels=None, *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> list[dict[str, Any]]:
    _candidate_config(candidate_key)
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), candidate_key=candidate_key, force_fallback=force_fallback) for shape in selected]

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str) -> list[dict[str, Any]]:
    new_seed_ids = set(_candidate_config(candidate_key)['new_seed_ids'])
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
        selected_seed = out.get('selected_seed')
        expected_seed = out.get('expected_seed')
        selected_new_seed = selected_seed in new_seed_ids
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_1877_kernel_ms'] = baseline_ms
        out['shape_specific_kernel_ms'] = candidate_ms if selected_new_seed else None
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['relative_speedup_vs_1877'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_1877'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        if expected_seed in new_seed_ids and selected_seed != expected_seed:
            out['classification'] = 'guard-miss'
        elif speedup_vs_external is not None and speedup_vs_external < 1.05:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        elif selected_new_seed and speedup_vs_baseline is not None and (speedup_vs_baseline < 1.0):
            out['classification'] = 'kernel-slow'
        elif selected_new_seed:
            out['classification'] = 'seed-consumed'
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
        trace_row = trace_by_label.get(label, {})
        classification = trace_row.get('classification', 'unmeasured')
        if classification in ('route-ok', 'unmeasured') and (not trace_row.get('selected_seed')):
            classification = 'fallback-slow'
        elif classification in ('route-ok', 'unmeasured') and trace_row.get('selected_seed'):
            classification = 'kernel-slow'
        rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'selected_seed': trace_row.get('selected_seed'), 'expected_seed': trace_row.get('expected_seed'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': classification})
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
        matrix.append({'shape_key': label, 'baseline_route': _base_route(inputs), 'candidate_route': route_for_contract_inputs(inputs, candidate_key=candidate_key), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'baseline_1877_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_1877': candidate_ms - baseline_ms if candidate_ms is not None and baseline_ms is not None else None, 'speedup_vs_1877': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_row': TARGETED_SEED_ROWS.get(selected_seed, {}), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def benchmark_baseline_1877(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_parent_1877_full90_v1, correctness=benchmark_correctness, time_flashlib=time_flashlib)

def _baseline_sidecar(report: dict[str, Any], *, shape_labels, denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    route_trace = route_trace_for_contract_shapes(shape_labels, candidate_key=BASE_1877_KEY)
    below_1x = _below_flashlib_rows(report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(report, route_trace, floor=1.05)
    return {'candidate_id': BASE_1877_ID, 'candidate_key': BASE_1877_KEY, 'selected_seeds': CANDIDATE_CONFIGS[BASE_1877_KEY]['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': BASE_1877_ENTRYPOINT, 'route_entrypoint': BASE_1877_ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'route_trace': route_trace, 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': _timing_backends_for_reports(report), 'timing_backend_requested': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'report': report}

def _benchmark_payload(candidate_key: str, candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key), candidate_report, baseline_report, candidate_key=candidate_key)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=1.05)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    config = _candidate_config(candidate_key)
    return {'candidate_id': config['candidate_id'], 'candidate_key': candidate_key, 'baseline_candidate_id': BASE_1877_ID, 'selected_seeds': config['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_1877_tflops': baseline_metric, 'metric_delta_vs_1877': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': config['benchmark_entrypoint'], 'baseline_entrypoint': BASE_1877_ENTRYPOINT, 'route_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': config['expected_shape_wins'], 'selected_route_rows': _rows_for_labels(candidate_report, config['expected_shape_wins']), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, config['expected_shape_wins']), 'seed_delta_matrix': _seed_delta_matrix(candidate_key, candidate_report, baseline_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'rejected_route_combinations': REJECTED_ROUTE_COMBINATIONS, 'selected_candidate_dispatcher': config['candidate_id'], 'guard_plan': config['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'speedup_floor': 1.05, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_1877_value': baseline_metric, 'delta_vs_1877': candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_portfolio(candidate_key: str, *, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    if candidate_key == BASE_1877_KEY:
        baseline = benchmark_baseline_1877(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
        return _baseline_sidecar(baseline, shape_labels=shape_labels, denominator=_denominator_name(shape_labels), timing_backend=_timing_backend_name(use_cupti), benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    if baseline_report is None:
        baseline_report = benchmark_baseline_1877(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key), correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_key, candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)

def benchmark_candidate_9a17_only_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_9A17_ONLY, **kwargs)

def benchmark_candidate_b644_only_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_B644_ONLY, **kwargs)

def benchmark_candidate_6a35_d64_only_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_6A35_D64_ONLY, **kwargs)

def benchmark_candidate_b644_6a35_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_B644_6A35, **kwargs)

def benchmark_candidate_9a17_b644_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_9A17_B644, **kwargs)

def benchmark_candidate_9a17_6a35_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_9A17_6A35, **kwargs)

def benchmark_candidate_9a17_b644_6a35_full90_v1(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(CANDIDATE_9A17_B644_6A35, **kwargs)

def _candidate_no_regresses_baseline(payload: dict[str, Any], baseline_value: float | None) -> bool:
    value = payload.get('tflops')
    return payload.get('all_correct') and payload.get('performance_comparable') and (value is not None) and (baseline_value is None or value >= baseline_value)

def _best_candidate_key(payloads: dict[str, dict[str, Any]]) -> str | None:
    baseline_value = payloads.get(BASE_1877_KEY, {}).get('tflops')
    candidates = [key for key, payload in payloads.items() if key != BASE_1877_KEY and _candidate_no_regresses_baseline(payload, baseline_value)]
    if not candidates:
        return None
    return max(candidates, key=lambda key: (payloads[key].get('tflops') or float('-inf'), len(CANDIDATE_CONFIGS[key]['new_seed_ids'])))

def _summary_payload(*, payloads: dict[str, dict[str, Any]], artifacts: dict[str, str], denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool) -> dict[str, Any]:
    selected_key = _best_candidate_key(payloads)
    selected_payload = payloads.get(selected_key, {}) if selected_key else {}
    baseline_payload = payloads[BASE_1877_KEY]
    return {'candidate_id': 'dispatcher_synthesis_1877_9a17_b644_6a35_full90_selective_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':write_benchmark_artifacts']), 'denominator': denominator, 'timing_backend': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'baseline_candidate_key': BASE_1877_KEY, 'selected_candidate_key': selected_key, 'selected_candidate_dispatcher': _candidate_id(selected_key) if selected_key else None, 'default_candidate_key': DEFAULT_CANDIDATE_KEY, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'rejected_route_combinations': REJECTED_ROUTE_COMBINATIONS, 'candidate_rankings': [{'candidate_key': key, 'candidate_id': payloads[key].get('candidate_id'), 'tflops': payloads[key].get('tflops'), 'metric_delta_vs_1877': payloads[key].get('metric_delta_vs_1877'), 'all_correct': payloads[key].get('all_correct'), 'performance_comparable': payloads[key].get('performance_comparable'), 'performance_coverage': payloads[key].get('performance_coverage')} for key in CANDIDATE_KEYS if key in payloads], 'seed_delta_matrix': selected_payload.get('seed_delta_matrix', []), 'seed_delta_matrix_all_candidates': {key: payloads[key].get('seed_delta_matrix', []) for key in payloads if key != BASE_1877_KEY}, 'flashlib_parity_ledger': selected_payload.get('flashlib_parity_ledger', baseline_payload.get('flashlib_parity_ledger', {})), 'full_denominator_ab': {'baseline_payload': artifacts.get('same_session_baseline_payload'), 'candidate_payload': artifacts.get(''.join([format(selected_key, ''), '_payload'])) if selected_key else None, 'denominator': denominator, 'timing_backend': timing_backend, 'metric_delta': selected_payload.get('metric_delta_vs_1877'), 'route_trace': selected_payload.get('route_trace', [])}, 'baseline_tflops': baseline_payload.get('tflops'), 'selected_tflops': selected_payload.get('tflops'), 'artifacts': artifacts}

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True, candidate_keys: tuple[str, ...] | None=None) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom_label = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    baseline_report = benchmark_baseline_1877(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_payload = _baseline_sidecar(baseline_report, shape_labels=shape_labels, denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    artifacts: dict[str, str] = {}
    payloads = {BASE_1877_KEY: baseline_payload}
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_1877_v1.json'])
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['same_session_baseline_payload'] = str(baseline_path)
    selected_candidate_keys = list(DEFAULT_SYNTHESIS_CANDIDATES) if candidate_keys is None else list(candidate_keys)
    for candidate_key in selected_candidate_keys:
        if candidate_key == BASE_1877_KEY:
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
    summary_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_synthesis_1877_9a17_b644_6a35_selective_v1.json'])
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    artifacts['dispatcher_synthesis'] = str(summary_path)
    return artifacts
