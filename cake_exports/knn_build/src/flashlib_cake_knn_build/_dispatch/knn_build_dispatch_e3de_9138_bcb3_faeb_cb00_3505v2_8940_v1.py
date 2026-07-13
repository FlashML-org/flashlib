"""8940 kNN build dispatcher consumption of the 9a11 Q16/K32 seed.

Minimum target architecture: sm_100a. This dispatcher-consumption wrapper
starts from the 2422 guard policy, keeps the exact cb00 Q1 M131071/M250000
guards, and replaces only the Q16/K32 RAG microbucket route with the 9a11
``knn_build_rag_microbucket_3505_v2`` seed. No seed schedule, tile pipeline, or
math semantics are retuned here.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_e3de_9138_bcb3_4247_v1 as base_4247
from . import knn_build_rag_microbucket_3505_v1 as rag_3505_v1
from . import knn_build_rag_microbucket_3505_v2 as rag_3505_v2
from . import knn_build_rag_microbucket_faeb_v1 as rag_faeb
from . import knn_build_ragonline_mbucket_cb00_q1m_v2 as q1_cb00
MODULE = 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v2_8940_v1'
ROUTE_BASE_4247 = 'loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs'
ROUTE_Q1_CB00_PARENT_SPLIT72 = 'rag_online_mbucket_cb00_q1m_v2_parent_split72'
ROUTE_Q1_CB00_CTA1_S144_G12 = 'rag_online_mbucket_cb00_q1m_v2_cta1_s144_g12'
ROUTE_RAG_Q4_FAEB = rag_faeb.ROUTE_Q4_K10
ROUTE_RAG_Q64_FAEB = rag_faeb.ROUTE_Q64_K10
ROUTE_RAG_Q16_K32_FAEB = rag_faeb.ROUTE_Q16_K32
ROUTE_RAG_Q16_K32_3505_V1 = rag_3505_v1.ROUTE_Q16_K32
ROUTE_RAG_Q16_K32_3505_V2 = ''.join(['rag_microbucket_3505_v2_q16_k32_tailinf_cta1_s', format(rag_3505_v2.K32_SPLIT_COUNT, ''), '_g', format(rag_3505_v2.K32_GROUP_COUNT, '')])
CB00_SEED_ID = 'q1_mbucket_cb00_q1m_v2_4444'
FAEB_SEED_ID = 'rag_microbucket_faeb_v1'
SEED_3505_V1_ID = 'rag_microbucket_3505_v1'
SEED_3505_V2_ID = 'rag_microbucket_3505_v2_9a11'
DEFAULT_PORTFOLIO_ID = base_4247.DEFAULT_PORTFOLIO_ID
D64_TARGET_SHAPES = base_4247.D64_TARGET_SHAPES
RECT_TARGET_SHAPES = base_4247.RECT_TARGET_SHAPES
Q1_TARGET_SHAPES = q1_cb00.TARGET_SHAPES
Q1_CB00_SELECTED_M = (131071, 250000)
Q1_CB00_SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10"]}'))
RAG_Q4_Q64_TARGET_SHAPES = rag_faeb.K10_TARGET_SHAPES
RAG_Q16_K32_TARGET_SHAPES = rag_faeb.K32_TARGET_SHAPES
RAG_MICROBUCKET_TARGET_SHAPES = rag_faeb.TARGET_SHAPES
K96_AUDIT_SHAPES = ('build_over64_stress_qm2048_k96',)
NEW_CONSUMED_SEED_TARGET_SHAPES = RAG_Q16_K32_TARGET_SHAPES
DISPATCH_DELTA_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "build_over64_stress_qm2048_k96"]}'))
CONSUMED_SEED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32"]}'))
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32"]}'))
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "flashml_correctness_b1_q256_m256_d128_k5", "build_over32_stress_qm2048_k64", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "build_k_sweep_qm512_k5", "build_over32_stress_qm4096_k64"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "rag_microbatch_b1_q4_m100000_d128_k10", "rag_microbatch_b1_q64_m100000_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_qm2048_d128_k10", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
ROUTE_SEED_ID = {**base_4247.ROUTE_SEED_ID, ROUTE_Q1_CB00_PARENT_SPLIT72: CB00_SEED_ID, ROUTE_Q1_CB00_CTA1_S144_G12: CB00_SEED_ID, ROUTE_RAG_Q4_FAEB: FAEB_SEED_ID, ROUTE_RAG_Q64_FAEB: FAEB_SEED_ID, ROUTE_RAG_Q16_K32_FAEB: FAEB_SEED_ID, ROUTE_RAG_Q16_K32_3505_V1: SEED_3505_V1_ID, ROUTE_RAG_Q16_K32_3505_V2: SEED_3505_V2_ID}
ROUTE_ENTRYPOINTS = {**base_4247.ROUTE_ENTRYPOINTS, ROUTE_Q1_CB00_PARENT_SPLIT72: 'loom.examples.weave.knn_build_ragonline_mbucket_cb00_q1m_v2:launch_from_contract_inputs', ROUTE_Q1_CB00_CTA1_S144_G12: 'loom.examples.weave.knn_build_ragonline_mbucket_cb00_q1m_v2:launch_from_contract_inputs', ROUTE_RAG_Q4_FAEB: 'loom.examples.weave.knn_build_rag_microbucket_faeb_v1:launch_from_contract_inputs', ROUTE_RAG_Q64_FAEB: 'loom.examples.weave.knn_build_rag_microbucket_faeb_v1:launch_from_contract_inputs', ROUTE_RAG_Q16_K32_FAEB: 'loom.examples.weave.knn_build_rag_microbucket_faeb_v1:launch_from_contract_inputs', ROUTE_RAG_Q16_K32_3505_V1: 'loom.examples.weave.knn_build_rag_microbucket_3505_v1:launch_from_contract_inputs', ROUTE_RAG_Q16_K32_3505_V2: 'loom.examples.weave.knn_build_rag_microbucket_3505_v2:launch_from_contract_inputs'}
PRODUCTION_ROUTE_MODULES = {**base_4247.PRODUCTION_ROUTE_MODULES, CB00_SEED_ID: 'loom.examples.weave.knn_build_ragonline_mbucket_cb00_q1m_v2:launch_from_contract_inputs', 'rag_microbucket_faeb_q4_q64_k10': 'loom.examples.weave.knn_build_rag_microbucket_faeb_v1:launch_from_contract_inputs', SEED_3505_V1_ID: 'loom.examples.weave.knn_build_rag_microbucket_3505_v1:launch_from_contract_inputs', SEED_3505_V2_ID: 'loom.examples.weave.knn_build_rag_microbucket_3505_v2:launch_from_contract_inputs', 'base_4247': ROUTE_BASE_4247}
CANDIDATE_DISPATCHERS = ({'id': 'baseline_6d62', 'entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_6d62']), 'consumed_seeds': ('d64_fdd7_aa88_v2', 'rect_d64_cf49_v3_9138', 'q1_mbucket_aa88_q1m_v3_bcb3', FAEB_SEED_ID), 'guard_plan': ('faeb exact Q4/Q64 K10', 'faeb exact Q16 K32', 'then inherited 4247 guard stack'), 'expected_shape_wins': RAG_MICROBUCKET_TARGET_SHAPES, 'fallback': ROUTE_BASE_4247, 'rejected_reason': 'same-session baseline'}, {'id': 'baseline_2422', 'entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_2422']), 'consumed_seeds': (CB00_SEED_ID, FAEB_SEED_ID, SEED_3505_V1_ID), 'guard_plan': ('cb00 exact Q1 M131071/M250000 bucket', 'faeb exact Q4/Q64 K10', '3505 v1 exact Q16 K32', 'then inherited 4247 guard stack'), 'expected_shape_wins': (*Q1_CB00_SELECTED_TARGET_SHAPES, *RAG_MICROBUCKET_TARGET_SHAPES), 'fallback': ROUTE_BASE_4247, 'rejected_reason': 'lane base before consuming 9a11'}, {'id': 'candidate_2422_9a11_8940_v1', 'entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v2_8940_v1']), 'consumed_seeds': (CB00_SEED_ID, FAEB_SEED_ID, SEED_3505_V2_ID), 'guard_plan': ('cb00 exact Q1 M131071/M250000 bucket', 'faeb exact Q4/Q64 K10', '3505 v2 exact Q16 K32', 'then inherited 4247 guard stack'), 'expected_shape_wins': (*Q1_CB00_SELECTED_TARGET_SHAPES, *RAG_MICROBUCKET_TARGET_SHAPES), 'fallback': ROUTE_BASE_4247, 'rejected_reason': None})
TARGETED_SEED_ROWS = {**base_4247.TARGETED_SEED_ROWS, 'rag_online_b1_q1_m100000_d128_k10': {'kernel_ms': 0.056641, 'flashlib_ms': 0.060833, 'ratio_vs_flashlib': 1.0740099927614273, 'split_count': 72, 'route': base_4247.ROUTE_Q1_BCB3_SPLIT72}, 'rag_online_irregular_b1_q1_m131071_d128_k10': {'kernel_ms': 0.048641, 'flashlib_ms': 0.067425, 'ratio_vs_flashlib': 1.386176271047059, 'split_count': 144, 'group_count': 12, 'route': ROUTE_Q1_CB00_CTA1_S144_G12}, 'rag_online_large_m_b1_q1_m250000_d128_k10': {'kernel_ms': 0.0752, 'flashlib_ms': 0.091521, 'ratio_vs_flashlib': 1.217034574468085, 'split_count': 144, 'group_count': 12, 'route': ROUTE_Q1_CB00_CTA1_S144_G12}, rag_faeb.Q4_K10_SHAPE: {'kernel_ms': 0.06205, 'flashlib_ms': 0.063041, 'ratio_vs_flashlib': 1.0159709911361805, 'split_count': rag_faeb.M64_SPLIT_COUNT, 'group_count': rag_faeb.M64_GROUP_COUNT, 'route': ROUTE_RAG_Q4_FAEB}, rag_faeb.Q64_K10_SHAPE: {'kernel_ms': 0.072737, 'flashlib_ms': 0.098977, 'ratio_vs_flashlib': 1.3607517494535106, 'split_count': rag_faeb.M64_SPLIT_COUNT, 'group_count': rag_faeb.M64_GROUP_COUNT, 'route': ROUTE_RAG_Q64_FAEB}, rag_faeb.Q16_K32_SHAPE: {'kernel_ms': 0.140545, 'flashlib_ms': 0.134497, 'ratio_vs_flashlib': 0.9569675192998683, 'split_count': rag_3505_v2.K32_SPLIT_COUNT, 'group_count': rag_3505_v2.K32_GROUP_COUNT, 'route': ROUTE_RAG_Q16_K32_3505_V2}}
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _eligible_q1_cb00(inputs: dict[str, Any]) -> bool:
    return q1_cb00._eligible_rag_online_mbucket(inputs)

def _selected_q1_cb00_mbucket(inputs: dict[str, Any], *, enable_q1_cb00_m100k: bool=False) -> bool:
    if not _eligible_q1_cb00(inputs):
        return False
    return enable_q1_cb00_m100k or int(inputs.get('M', -1)) in Q1_CB00_SELECTED_M

def _eligible_rag_q4_q64(inputs: dict[str, Any]) -> bool:
    return rag_faeb._eligible_q4_k10(inputs) or rag_faeb._eligible_q64_k10(inputs)

def _eligible_rag_q16_k32(inputs: dict[str, Any]) -> bool:
    return rag_faeb._eligible_q16_k32(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, portfolio_id: str=DEFAULT_PORTFOLIO_ID, force_fallback: bool=False, enable_d64_fdd7: bool=True, enable_rect_d64: bool=True, enable_q1_mbucket: bool=True, enable_q1_cb00: bool=True, enable_q1_cb00_m100k: bool=False, enable_rag_microbucket: bool=True, enable_rag_q16_k32: bool=True, enable_rag_3505_v1_q16: bool=True, enable_rag_3505_v2_q16: bool=True, enable_rag_seed_portfolio: bool=True, enable_q512_k456: bool=True) -> str:
    if not force_fallback:
        if enable_q1_mbucket and enable_q1_cb00 and _selected_q1_cb00_mbucket(inputs, enable_q1_cb00_m100k=enable_q1_cb00_m100k):
            return q1_cb00.route_for_contract_inputs(inputs)
        if enable_rag_microbucket and _eligible_rag_q4_q64(inputs):
            return rag_faeb.route_for_contract_inputs(inputs)
        if enable_rag_microbucket and enable_rag_q16_k32 and _eligible_rag_q16_k32(inputs):
            if enable_rag_3505_v2_q16:
                return rag_3505_v2.route_for_contract_inputs(inputs)
            if enable_rag_3505_v1_q16:
                return rag_3505_v1.route_for_contract_inputs(inputs)
            return rag_faeb.route_for_contract_inputs(inputs)
    return base_4247.route_for_contract_inputs(inputs, portfolio_id=portfolio_id, force_fallback=force_fallback, enable_d64_fdd7=enable_d64_fdd7, enable_rect_d64=enable_rect_d64, enable_q1_mbucket=enable_q1_mbucket, enable_rag_seed_portfolio=enable_rag_seed_portfolio, enable_q512_k456=enable_q512_k456)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route in (ROUTE_Q1_CB00_PARENT_SPLIT72, ROUTE_Q1_CB00_CTA1_S144_G12):
        q1_cb00.launch_from_contract_inputs(inputs)
        return
    if route in (ROUTE_RAG_Q4_FAEB, ROUTE_RAG_Q64_FAEB, ROUTE_RAG_Q16_K32_FAEB):
        rag_faeb.launch_from_contract_inputs(inputs)
        return
    if route == ROUTE_RAG_Q16_K32_3505_V1:
        rag_3505_v1.launch_from_contract_inputs(inputs)
        return
    if route == ROUTE_RAG_Q16_K32_3505_V2:
        rag_3505_v2.launch_from_contract_inputs(inputs)
        return
    base_4247._launch_route(inputs, route)

def launch_from_contract_inputs(inputs: dict[str, Any], *, portfolio_id: str=DEFAULT_PORTFOLIO_ID, force_fallback: bool=False, enable_d64_fdd7: bool=True, enable_rect_d64: bool=True, enable_q1_mbucket: bool=True, enable_q1_cb00: bool=True, enable_q1_cb00_m100k: bool=False, enable_rag_microbucket: bool=True, enable_rag_q16_k32: bool=True, enable_rag_3505_v1_q16: bool=True, enable_rag_3505_v2_q16: bool=True, enable_rag_seed_portfolio: bool=True, enable_q512_k456: bool=True) -> None:
    route = route_for_contract_inputs(inputs, portfolio_id=portfolio_id, force_fallback=force_fallback, enable_d64_fdd7=enable_d64_fdd7, enable_rect_d64=enable_rect_d64, enable_q1_mbucket=enable_q1_mbucket, enable_q1_cb00=enable_q1_cb00, enable_q1_cb00_m100k=enable_q1_cb00_m100k, enable_rag_microbucket=enable_rag_microbucket, enable_rag_q16_k32=enable_rag_q16_k32, enable_rag_3505_v1_q16=enable_rag_3505_v1_q16, enable_rag_3505_v2_q16=enable_rag_3505_v2_q16, enable_rag_seed_portfolio=enable_rag_seed_portfolio, enable_q512_k456=enable_q512_k456)
    _launch_route(inputs, route)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_baseline_2422(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_rag_3505_v2_q16=False)

def candidate_baseline_6d62(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_q1_cb00=False, enable_rag_3505_v1_q16=False, enable_rag_3505_v2_q16=False)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base_4247._select_contract_shapes(shape_labels)

def compile_and_launch_knn_build(*, shape_labels=DISPATCH_CORRECTNESS_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    prior_use_cupti = _set_bench_backend(use_cupti)
    try:
        shapes = None if shape_labels is None else _select_contract_shapes(shape_labels)
        return evaluate_contract(shapes=shapes, correctness=True, benchmark=True, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = prior_use_cupti

def _trace_inputs_from_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return base_4247._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return base_4247._inputs_for_label(label)

def _selected_entrypoint_for_route(route: str) -> str:
    return ROUTE_ENTRYPOINTS.get(route, base_4247._selected_entrypoint_for_route(route))

def _base_route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    row = dict(base_4247._route_trace_record(inputs, force_fallback=force_fallback))
    route = str(row.get('selected_route') or base_4247.route_for_contract_inputs(inputs, force_fallback=force_fallback))
    row.setdefault('shape_key', inputs.get('label'))
    row.setdefault('selected_entrypoint', _selected_entrypoint_for_route(route))
    row.setdefault('selected_seed', row.get('selected_seed') or row.get('consumed_seed'))
    row.setdefault('expected_seed', row.get('selected_seed'))
    row.setdefault('route_kind', row.get('route_kind', 'general'))
    row.setdefault('route_source', 'broad-dispatcher')
    row.setdefault('guard_id', row.get('candidate_guard_status'))
    row.setdefault('guard_condition', 'inherited 4247 guard plan')
    row.setdefault('classification', 'route-ok')
    row.setdefault('dispatcher_kernel_ms', None)
    row.setdefault('shape_specific_kernel_ms', None)
    row.setdefault('relative_speedup_vs_baseline', None)
    row['base_4247_route'] = base_4247.route_for_contract_inputs(inputs)
    return row

def _q1_cb00_trace_record(inputs: dict[str, Any], *, base_4247_route: str) -> dict[str, Any]:
    label = str(inputs.get('label'))
    targeted = dict(TARGETED_SEED_ROWS[label])
    route = q1_cb00.route_for_contract_inputs(inputs)
    return {'shape_key': inputs.get('label'), 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINTS[route], 'selected_seed': CB00_SEED_ID, 'expected_seed': CB00_SEED_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': 'q1_online_mbucket_cb00_q1m_v2_selected_m_exact', 'guard_condition': ''.join(['exact BF16 non-build B=1 Q=1 M=', format(int(inputs.get('M')), ''), ' D=128 K=10 cb00 Q1 online M-bucket seed']), 'coverage': '8940 keeps 2422 cb00 Q1 guard ahead of inherited 4247 bcb3', 'consumed_seed': CB00_SEED_ID, 'replaced_route': base_4247_route, 'base_4247_route': base_4247_route, 'row_selection': targeted, 'split_count': targeted['split_count'], 'group_count': targeted.get('group_count'), 'targeted_seed_timing_backend': 'cupti', 'targeted_seed_kernel_ms': targeted['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': targeted['ratio_vs_flashlib'], 'classification': 'seed-consumed', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': targeted['kernel_ms'], 'relative_speedup_vs_baseline': None}

def _rag_microbucket_trace_record(inputs: dict[str, Any], *, route: str, base_4247_route: str) -> dict[str, Any]:
    label = str(inputs.get('label'))
    targeted = dict(TARGETED_SEED_ROWS[label])
    if route == ROUTE_RAG_Q4_FAEB:
        seed = FAEB_SEED_ID
        guard_id = 'rag_microbucket_faeb_q4_k10_exact'
        guard_condition = 'exact BF16 non-build B=1 Q=4 M=100000 D=128 K=10 faeb seed'
    elif route == ROUTE_RAG_Q64_FAEB:
        seed = FAEB_SEED_ID
        guard_id = 'rag_microbucket_faeb_q64_k10_exact'
        guard_condition = 'exact BF16 non-build B=1 Q=64 M=100000 D=128 K=10 faeb seed'
    elif route == ROUTE_RAG_Q16_K32_3505_V2:
        seed = SEED_3505_V2_ID
        guard_id = 'rag_microbucket_3505_v2_q16_k32_exact'
        guard_condition = 'exact BF16 non-build B=1 Q=16 M=100000 D=128 K=32 9a11 3505_v2 seed'
    elif route == ROUTE_RAG_Q16_K32_3505_V1:
        seed = SEED_3505_V1_ID
        guard_id = 'rag_microbucket_3505_v1_q16_k32_exact'
        guard_condition = 'exact BF16 non-build B=1 Q=16 M=100000 D=128 K=32 2422 3505_v1 seed'
    else:
        seed = FAEB_SEED_ID
        guard_id = 'rag_microbucket_faeb_q16_k32_exact'
        guard_condition = 'exact BF16 non-build B=1 Q=16 M=100000 D=128 K=32 faeb seed'
    return {'shape_key': inputs.get('label'), 'selected_route': route, 'selected_entrypoint': ROUTE_ENTRYPOINTS[route], 'selected_seed': seed, 'expected_seed': seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': guard_id, 'guard_condition': guard_condition, 'coverage': '8940 consumes exact RAG microbucket seeds ahead of inherited 4247 fallback', 'consumed_seed': seed, 'replaced_route': base_4247_route, 'base_4247_route': base_4247_route, 'row_selection': targeted, 'split_count': targeted['split_count'], 'group_count': targeted.get('group_count'), 'targeted_seed_timing_backend': 'cupti', 'targeted_seed_kernel_ms': targeted['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': targeted['ratio_vs_flashlib'], 'classification': 'seed-consumed', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': targeted['kernel_ms'], 'relative_speedup_vs_baseline': None}

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False, enable_q1_cb00: bool=True, enable_q1_cb00_m100k: bool=False, enable_rag_3505_v1_q16: bool=True, enable_rag_3505_v2_q16: bool=True) -> dict[str, Any]:
    base_4247_route = base_4247.route_for_contract_inputs(inputs)
    is_overlay = _eligible_q1_cb00(inputs) or _eligible_rag_q4_q64(inputs) or _eligible_rag_q16_k32(inputs)
    if force_fallback and is_overlay:
        row = _base_route_trace_record(inputs, force_fallback=True)
        expected_route = route_for_contract_inputs(inputs, force_fallback=False, enable_q1_cb00=enable_q1_cb00, enable_q1_cb00_m100k=enable_q1_cb00_m100k, enable_rag_3505_v1_q16=enable_rag_3505_v1_q16, enable_rag_3505_v2_q16=enable_rag_3505_v2_q16)
        row['selected_route'] = base_4247.route_for_contract_inputs(inputs, force_fallback=True)
        row['selected_entrypoint'] = _selected_entrypoint_for_route(str(row['selected_route']))
        row['selected_seed'] = row.get('consumed_seed')
        row['expected_seed'] = ROUTE_SEED_ID.get(expected_route)
        row['guard_id'] = 'forced_fallback_8940_overlays_disabled'
        row['guard_condition'] = 'forced fallback to inherited 4247 path; 8940 overlays disabled'
        row['forced_disabled_seeds'] = (CB00_SEED_ID, FAEB_SEED_ID, SEED_3505_V1_ID, SEED_3505_V2_ID)
        row['candidate_guard_status'] = 'forced_fallback'
        row['classification'] = 'route-ok'
        return row
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback, enable_q1_cb00=enable_q1_cb00, enable_q1_cb00_m100k=enable_q1_cb00_m100k, enable_rag_3505_v1_q16=enable_rag_3505_v1_q16, enable_rag_3505_v2_q16=enable_rag_3505_v2_q16)
    if route in (ROUTE_Q1_CB00_PARENT_SPLIT72, ROUTE_Q1_CB00_CTA1_S144_G12):
        return _q1_cb00_trace_record(inputs, base_4247_route=base_4247_route)
    if route in (ROUTE_RAG_Q4_FAEB, ROUTE_RAG_Q64_FAEB, ROUTE_RAG_Q16_K32_FAEB, ROUTE_RAG_Q16_K32_3505_V1, ROUTE_RAG_Q16_K32_3505_V2):
        return _rag_microbucket_trace_record(inputs, route=route, base_4247_route=base_4247_route)
    return _base_route_trace_record(inputs, force_fallback=force_fallback)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False, enable_q1_cb00: bool=True, enable_q1_cb00_m100k: bool=False, enable_rag_3505_v1_q16: bool=True, enable_rag_3505_v2_q16: bool=True) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback, enable_q1_cb00=enable_q1_cb00, enable_q1_cb00_m100k=enable_q1_cb00_m100k, enable_rag_3505_v1_q16=enable_rag_3505_v1_q16, enable_rag_3505_v2_q16=enable_rag_3505_v2_q16) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return base_4247._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return base_4247._rows_for_labels(report, labels)

def _row_delta(label: str, route: str, candidate_report: dict[str, Any], baseline_2422_report: dict[str, Any], baseline_6d62_report: dict[str, Any], *, candidate_id: str) -> dict[str, Any]:
    candidate_row = candidate_report.get('per_shape', {}).get(label, {})
    baseline_2422_row = baseline_2422_report.get('per_shape', {}).get(label, {})
    baseline_6d62_row = baseline_6d62_report.get('per_shape', {}).get(label, {})
    candidate_ms = candidate_row.get('kernel_ms')
    baseline_2422_ms = baseline_2422_row.get('kernel_ms')
    baseline_6d62_ms = baseline_6d62_row.get('kernel_ms')
    flashlib_ms = candidate_row.get('flashlib_ms')
    return {'candidate_id': candidate_id, 'selected_seed': ROUTE_SEED_ID.get(route), 'selected_route': route, 'candidate_ms': candidate_ms, 'baseline_2422_ms': baseline_2422_ms, 'baseline_6d62_ms': baseline_6d62_ms, 'flashlib_ms': flashlib_ms, 'metric_delta': candidate_ms - baseline_2422_ms if candidate_ms and baseline_2422_ms else None, 'speedup_vs_2422': baseline_2422_ms / candidate_ms if candidate_ms and baseline_2422_ms else None, 'speedup_vs_6d62': baseline_6d62_ms / candidate_ms if candidate_ms and baseline_6d62_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'timing_backend': candidate_row.get('timing_backend') or baseline_2422_row.get('timing_backend') or baseline_6d62_row.get('timing_backend') or 'cupti'}

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_2422_report: dict[str, Any], baseline_6d62_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in DISPATCH_DELTA_SHAPES:
        inputs = _inputs_for_label(label)
        baseline_6d62_route = route_for_contract_inputs(inputs, enable_q1_cb00=False, enable_rag_3505_v1_q16=False, enable_rag_3505_v2_q16=False)
        baseline_2422_route = route_for_contract_inputs(inputs, enable_rag_3505_v2_q16=False)
        candidate_route = route_for_contract_inputs(inputs)
        matrix.append({'shape_key': label, 'baseline_route': baseline_2422_route, 'baseline_6d62_route': baseline_6d62_route, 'candidate_deltas': [_row_delta(label, baseline_6d62_route, baseline_6d62_report, baseline_2422_report, baseline_6d62_report, candidate_id='baseline_6d62'), _row_delta(label, baseline_2422_route, baseline_2422_report, baseline_2422_report, baseline_6d62_report, candidate_id='baseline_2422'), _row_delta(label, candidate_route, candidate_report, baseline_2422_report, baseline_6d62_report, candidate_id='candidate_2422_9a11_8940_v1')]})
    return matrix

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_2422_report: dict[str, Any], baseline_6d62_report: dict[str, Any]) -> dict[str, Any]:
    deltas = {}
    for item in _seed_delta_matrix(candidate_report, baseline_2422_report, baseline_6d62_report):
        label = item['shape_key']
        candidate_delta = item['candidate_deltas'][-1]
        deltas[label] = {'candidate_ms': candidate_delta.get('candidate_ms'), 'baseline_2422_ms': candidate_delta.get('baseline_2422_ms'), 'baseline_6d62_ms': candidate_delta.get('baseline_6d62_ms'), 'flashlib_ms': candidate_delta.get('flashlib_ms'), 'speedup_vs_2422': candidate_delta.get('speedup_vs_2422'), 'speedup_vs_6d62': candidate_delta.get('speedup_vs_6d62'), 'ratio_vs_flashlib': candidate_delta.get('ratio_vs_flashlib'), 'candidate_route': candidate_delta.get('selected_route'), 'baseline_2422_route': item['baseline_route'], 'baseline_6d62_route': item['baseline_6d62_route'], 'selected_seed': candidate_delta.get('selected_seed'), 'targeted_seed_kernel_ms': TARGETED_SEED_ROWS.get(label, {}).get('kernel_ms'), 'targeted_seed_ratio_vs_flashlib': TARGETED_SEED_ROWS.get(label, {}).get('ratio_vs_flashlib'), 'candidate_passed': candidate_report.get('per_shape', {}).get(label, {}).get('passed'), 'baseline_2422_passed': baseline_2422_report.get('per_shape', {}).get(label, {}).get('passed'), 'baseline_6d62_passed': baseline_6d62_report.get('per_shape', {}).get(label, {}).get('passed')}
    return deltas

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_2422_report: dict[str, Any], baseline_6d62_report: dict[str, Any]) -> list[dict[str, Any]]:
    annotated = []
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_2422_row = baseline_2422_report.get('per_shape', {}).get(label, {})
        baseline_6d62_row = baseline_6d62_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_2422_ms = baseline_2422_row.get('kernel_ms')
        baseline_6d62_ms = baseline_6d62_row.get('kernel_ms')
        ratio = candidate_row.get('ratio_vs_flashlib')
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_2422_dispatcher_kernel_ms'] = baseline_2422_ms
        out['baseline_6d62_dispatcher_kernel_ms'] = baseline_6d62_ms
        out['flashlib_ms'] = candidate_row.get('flashlib_ms')
        out['relative_speedup_vs_baseline'] = baseline_2422_ms / candidate_ms if candidate_ms and baseline_2422_ms else None
        out['relative_speedup_vs_6d62'] = baseline_6d62_ms / candidate_ms if candidate_ms and baseline_6d62_ms else None
        out['route_changed_vs_baseline_2422'] = out.get('selected_route') != route_for_contract_inputs(_inputs_for_label(label), enable_rag_3505_v2_q16=False)
        if label in NEW_CONSUMED_SEED_TARGET_SHAPES and out.get('selected_seed') == SEED_3505_V2_ID:
            if isinstance(ratio, (float, int)) and ratio < 1.0:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'seed-consumed'
        elif not out['route_changed_vs_baseline_2422']:
            if isinstance(ratio, (float, int)) and ratio < 1.0:
                out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
            else:
                out['classification'] = 'route-ok'
        elif label in DISPATCH_DELTA_SHAPES and out.get('route_changed_vs_baseline_2422'):
            speedup = out['relative_speedup_vs_baseline']
            out['classification'] = 'seed-consumed' if speedup is None or speedup >= 1.0 else 'kernel-slow'
        elif isinstance(ratio, (float, int)) and ratio < 1.0:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        else:
            out['classification'] = out.get('classification', 'route-ok')
        annotated.append(out)
    return annotated

def _below_flashlib_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace_for_contract_shapes()}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < 1.0:
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': 'kernel-slow' if trace_row.get('route_kind') == 'specialized' else 'fallback-slow'})
    return rows

def _hot_bucket_parity(report: dict[str, Any]) -> dict[str, str]:
    buckets = {'d64_build_q1024_q2048_q4096_k10': D64_TARGET_SHAPES, 'rectangular_search_q1024_m32768_d64_k10': RECT_TARGET_SHAPES, 'rag_online_q1_mbucket': Q1_TARGET_SHAPES, 'rag_microbatch_q4_q64_k10': RAG_Q4_Q64_TARGET_SHAPES, 'rag_microbatch_q16_k32': RAG_Q16_K32_TARGET_SHAPES, 'build_over64_k96': K96_AUDIT_SHAPES}
    out = {}
    for bucket, labels in buckets.items():
        out[bucket] = 'pass'
        for label in labels:
            ratio = report.get('per_shape', {}).get(label, {}).get('ratio_vs_flashlib')
            if not isinstance(ratio, (float, int)) or ratio < 1.0:
                out[bucket] = 'fail'
                break
    return out

def _benchmark_payload(candidate_report: dict[str, Any], baseline_2422_report: dict[str, Any], baseline_6d62_report: dict[str, Any], *, use_cupti: bool, shape_labels, measured_function: str, candidate_id: str) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_2422_metric = baseline_2422_report['summary']['primary_mean']
    baseline_6d62_metric = baseline_6d62_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels), candidate_report, baseline_2422_report, baseline_6d62_report)
    below_flashlib = _below_flashlib_rows(candidate_report)
    return {'candidate_id': candidate_id, 'tflops': candidate_metric, 'baseline_2422_tflops': baseline_2422_metric, 'baseline_6d62_tflops': baseline_6d62_metric, 'metric_delta_vs_2422': candidate_metric - baseline_2422_metric if candidate_metric and baseline_2422_metric else None, 'metric_delta_vs_6d62': candidate_metric - baseline_6d62_metric if candidate_metric and baseline_6d62_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_2422_all_correct': baseline_2422_report['summary']['all_correct'], 'baseline_6d62_all_correct': baseline_6d62_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_2422_performance_comparable': baseline_2422_report['summary']['performance_comparable'], 'baseline_6d62_performance_comparable': baseline_6d62_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':', format(measured_function, '')]), 'baseline_2422_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_2422']), 'baseline_6d62_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_6d62']), 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': SELECTED_TARGET_SHAPES, 'consumed_seed_labels': CONSUMED_SEED_TARGET_SHAPES, 'new_consumed_seed_labels': NEW_CONSUMED_SEED_TARGET_SHAPES, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, SELECTED_TARGET_SHAPES), 'baseline_2422_selected_route_rows': _rows_for_labels(baseline_2422_report, SELECTED_TARGET_SHAPES), 'baseline_6d62_selected_route_rows': _rows_for_labels(baseline_6d62_report, SELECTED_TARGET_SHAPES), 'consumed_seed_rows': _rows_for_labels(candidate_report, CONSUMED_SEED_TARGET_SHAPES), 'new_consumed_seed_rows': _rows_for_labels(candidate_report, NEW_CONSUMED_SEED_TARGET_SHAPES), 'baseline_2422_consumed_seed_rows': _rows_for_labels(baseline_2422_report, CONSUMED_SEED_TARGET_SHAPES), 'baseline_6d62_consumed_seed_rows': _rows_for_labels(baseline_6d62_report, CONSUMED_SEED_TARGET_SHAPES), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_2422_report, baseline_6d62_report), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_2422_report, baseline_6d62_report), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': 'candidate_2422_9a11_8940_v1', 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_2422_contract_summary': baseline_2422_report['summary'], 'baseline_6d62_contract_summary': baseline_6d62_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_2422_contract_performance': baseline_2422_report['performance'], 'baseline_6d62_contract_performance': baseline_6d62_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_2422_report, baseline_6d62_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'hot_bucket_parity': _hot_bucket_parity(candidate_report), 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_2422_report': baseline_2422_report, 'baseline_6d62_report': baseline_6d62_report}

def _benchmark_candidate(*, use_cupti: bool=True, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, measured_function: str, candidate_id: str, baseline_2422_report: dict[str, Any] | None=None, baseline_6d62_report: dict[str, Any] | None=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn)
    if baseline_2422_report is None:
        baseline_2422_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_2422)
    if baseline_6d62_report is None:
        baseline_6d62_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_6d62)
    return _benchmark_payload(candidate_report, baseline_2422_report, baseline_6d62_report, use_cupti=use_cupti, shape_labels=shape_labels, measured_function=measured_function, candidate_id=candidate_id)

def benchmark_knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v2_8940_v1(*, use_cupti: bool=True, shape_labels=None, baseline_2422_report: dict[str, Any] | None=None, baseline_6d62_report: dict[str, Any] | None=None) -> dict[str, Any]:
    return _benchmark_candidate(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate, measured_function='benchmark_knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v2_8940_v1', candidate_id='candidate_2422_9a11_8940_v1', baseline_2422_report=baseline_2422_report, baseline_6d62_report=baseline_6d62_report)

def benchmark_baseline_2422(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_2422)
    return {'candidate_id': 'baseline_2422', 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_2422']), 'route_trace': route_trace_for_contract_shapes(shape_labels, enable_rag_3505_v2_q16=False), 'route_trace_included': True, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report, 'contract_summary': report['summary'], 'contract_performance': report['performance']}

def benchmark_baseline_6d62(*, use_cupti: bool=True, shape_labels=None) -> dict[str, Any]:
    report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_6d62)
    return {'candidate_id': 'baseline_6d62', 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_baseline_6d62']), 'route_trace': route_trace_for_contract_shapes(shape_labels, enable_q1_cb00=False, enable_rag_3505_v1_q16=False, enable_rag_3505_v2_q16=False), 'route_trace_included': True, 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'report': report, 'contract_summary': report['summary'], 'contract_performance': report['performance']}

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None) -> dict[str, str]:
    payload = benchmark_knn_build_dispatch_e3de_9138_bcb3_faeb_cb00_3505v2_8940_v1(use_cupti=use_cupti, shape_labels=shape_labels)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_e3de_9138_bcb3_faeb_cb00_3505v2_8940_v1.json'])
    baseline_2422_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_2422_for_8940_v1.json'])
    baseline_6d62_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_6d62_for_8940_v1.json'])
    route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_e3de_9138_bcb3_faeb_cb00_3505v2_8940_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_e3de_9138_bcb3_faeb_cb00_3505v2_8940_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom, ''), '_seed_delta_matrix_e3de_9138_bcb3_faeb_cb00_3505v2_8940_v1.json'])
    candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    baseline_2422_path.write_text(json.dumps({'candidate_id': 'baseline_2422', 'measured_entrypoint': payload['baseline_2422_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_2422_tflops'], 'all_correct': payload['baseline_2422_all_correct'], 'performance_comparable': payload['baseline_2422_performance_comparable'], 'contract_summary': payload['baseline_2422_contract_summary'], 'contract_performance': payload['baseline_2422_contract_performance'], 'route_trace': route_trace_for_contract_shapes(shape_labels, enable_rag_3505_v2_q16=False), 'route_trace_included': True, 'report': payload['baseline_2422_report']}, indent=2, sort_keys=True) + '\n')
    baseline_6d62_path.write_text(json.dumps({'candidate_id': 'baseline_6d62', 'measured_entrypoint': payload['baseline_6d62_entrypoint'], 'measured_shape_labels': payload['measured_shape_labels'], 'timing_backend_requested': payload['timing_backend_requested'], 'timing_backends': payload['timing_backends'], 'tflops': payload['baseline_6d62_tflops'], 'all_correct': payload['baseline_6d62_all_correct'], 'performance_comparable': payload['baseline_6d62_performance_comparable'], 'contract_summary': payload['baseline_6d62_contract_summary'], 'contract_performance': payload['baseline_6d62_contract_performance'], 'route_trace': route_trace_for_contract_shapes(shape_labels, enable_q1_cb00=False, enable_rag_3505_v1_q16=False, enable_rag_3505_v2_q16=False), 'route_trace_included': True, 'report': payload['baseline_6d62_report']}, indent=2, sort_keys=True) + '\n')
    route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
    forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
    seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(candidate_path), 'same_session_baseline_2422_payload': str(baseline_2422_path), 'same_session_baseline_6d62_payload': str(baseline_6d62_path), 'route_trace': str(route_trace_path), 'forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path)}
