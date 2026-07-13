"""Full90 dispatcher consumption for the fd37 FP16 D128 K10 seed.

Minimum target architecture: sm_100a. This additive dispatcher wrapper keeps
the fd37-selected 1877+9a17 full90 route as the baseline, then routes only
``build_dtype_fp16_b1_q2048_m2048_d128_k10`` to the validated fd37 FP16
split8 cached-merge Weave seed.

Production routes stay Weave-only; FlashLib is timed only by the contract
harness as an external reference.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from . import knn_build_fp16_d128_lowfloor_fd37_v1 as fp16_fd37
from . import knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1 as parent_full90
MODULE = 'loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1'
eval_mod = parent_full90.eval_mod
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))
BASE_CANDIDATE_KEY = parent_full90.CANDIDATE_9A17_ONLY
BASE_CONFIG = parent_full90.CANDIDATE_CONFIGS[BASE_CANDIDATE_KEY]
BASE_CANDIDATE_ID = BASE_CONFIG['candidate_id']
BASE_BENCHMARK_ENTRYPOINT = BASE_CONFIG['benchmark_entrypoint']
BASE_ROUTE_ENTRYPOINT = BASE_CONFIG['entrypoint']
CANDIDATE_FP16 = '1877_plus_9a17_fp16_fd37_lowfloor'
DEFAULT_CANDIDATE_KEY = CANDIDATE_FP16
CANDIDATE_KEYS = (BASE_CANDIDATE_KEY, CANDIDATE_FP16)
CANDIDATE_ID = 'candidate_1877_plus_9a17_fp16_fd37_lowfloor_full90_v1'
ROUTE_ENTRYPOINT = ''.join([format(MODULE, ''), ':launch_from_contract_inputs'])
BENCHMARK_ENTRYPOINT = ''.join([format(MODULE, ''), ':benchmark_candidate_fp16_fd37_full90_v1'])
TARGET_FP16_D128_K10 = fp16_fd37.TARGET_SHAPE
TARGET_SHAPES = (TARGET_FP16_D128_K10,)
SEED_FP16_D128_K10_ID = 'fp16_d128_lowfloor_fd37_s8_cached_merge'
ROUTE_FP16_D128_K10_SEED = fp16_fd37.ROUTE_FP16_S8_CACHED_MERGE
FP16_SEED_ENTRYPOINT = 'loom.examples.weave.knn_build_fp16_d128_lowfloor_fd37_v1:launch_from_contract_inputs'
TARGETED_SEED_PAYLOAD = 'artifacts/weave_evolve/knn_build_fp16_d128_lowfloor_fd37_v1/fp16_d128_lowfloor_fd37_v1.json'
SOURCE_TASKS = {**parent_full90.SOURCE_TASKS, SEED_FP16_D128_K10_ID: 'weave-evolve-knn-build-7734 / design_doc/active/weave_evolve_knn_build_round_155_fd37_fp16.md', CANDIDATE_ID: 'generalize-auto-tuning-knn-build-fd9b dispatcher-consumption'}
PRODUCTION_ROUTE_MODULES = {**parent_full90.PRODUCTION_ROUTE_MODULES, SEED_FP16_D128_K10_ID: FP16_SEED_ENTRYPOINT, BASE_CANDIDATE_ID: BASE_ROUTE_ENTRYPOINT}
PARENT_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1"]}'))
SELECTED_SEEDS = _decode_capture(_json_loads('{"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge"]}'))
EXPECTED_SHAPE_WINS = _decode_capture(_json_loads('{"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10"]}'))
CANDIDATE_CONFIGS = _decode_capture(_json_loads('{"__dict_items__": [["1877_plus_9a17_residual_rag_search", {"__dict_items__": [["candidate_id", "candidate_1877_plus_9a17_residual_rag_search_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_9a17_only_full90_v1"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_9a17_only_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1"]}], ["new_seed_ids", {"__tuple__": []}], ["guard_plan", {"__tuple__": ["9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_9a17_only_full90_v1"], ["rejected_reason", "same-session fd37 selected 1877+9a17 full90 baseline"]]}], ["1877_plus_9a17_fp16_fd37_lowfloor", {"__dict_items__": [["candidate_id", "candidate_1877_plus_9a17_fp16_fd37_lowfloor_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:launch_from_contract_inputs"], ["benchmark_entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:benchmark_candidate_fp16_fd37_full90_v1"], ["selected_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge"]}], ["new_seed_ids", {"__tuple__": ["fp16_d128_lowfloor_fd37_s8_cached_merge"]}], ["guard_plan", {"__tuple__": ["fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_9a17_only_full90_v1"], ["rejected_reason", null]]}]]}'))
CANDIDATE_DISPATCHERS = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["id", "candidate_1877_plus_9a17_residual_rag_search_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:benchmark_candidate_9a17_only_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1"]}], ["guard_plan", {"__tuple__": ["9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_9a17_only_full90_v1"], ["rejected_reason", "same-session fd37 selected 1877+9a17 full90 baseline"]]}, {"__dict_items__": [["id", "candidate_1877_plus_9a17_fp16_fd37_lowfloor_full90_v1"], ["entrypoint", "loom.examples.weave.knn_build_dispatch_1877_9a17_fp16_fd37_full90_consumption_v1:benchmark_candidate_fp16_fd37_full90_v1"], ["consumed_seeds", {"__tuple__": ["q1024_k8_195e_v1", "ragonline_mbucket_5706_q1v10_smix_s144_g12", "rag_microbucket_k32_q16dual2warp_largem_bdd2_v1_rowld1_2warp_rows4", "rag_microbucket_k32_q24rowld2_24dc_v1_rowld2_rows4", "rag_stream_k32_q128rowld_60fb_v1_rowld_rows4", "4757_fixedbuild_k10_v2_q512_q1024_b2_q1536_q6144", "ad64_fixedbuild_k10_v2_q512_q1024_q1536_q2048_b2q1024", "rag_microbatch_k10_q8_s128_q16_s136_4757_v1", "rag_stream_k32_q128_m100000_ad64_v1_4fbf_v6_s72g8", "d15e_rect_smallq_largem_v1", "fp16_d128_lowfloor_fd37_s8_cached_merge"]}], ["guard_plan", {"__tuple__": ["fd37 exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached-merge guard", "9a17 exact BF16 non-build B=1 Q=128 M=100000 D=128 K=32 guard", "9a17 exact BF16 non-build B=1 Q=1024 M=8192 D=128 K=10 guard", "4b51 exact BF16 build K10 guard for unique Q2048 row", "1b8f exact BF16 build K10 guard for Q512/Q1024/B2-Q1024/Q1536/Q6144 rows", "ceb3 exact BF16 non-build B=1 Q in {8,16} M=100000 D=128 K=10 guard", "b0e2 exact BF16 non-build B=1 Q=128 M=131071 D=128 K=32 rowld rows4 guard", "603d/24dc exact BF16 non-build B=1 Q=24 M=100000 D=128 K=32 rowld2 rows4 guard", "cf51 exact BF16 build Q=M=1024 D=128 K=8 split16", "bca0 exact BF16 non-build Q=1 D=128 K=10 M in {65536,100000,131071,250000,262143,524287}", "5018 exact BF16 non-build Q=16 D=128 K=32 M in {100000,131071,250000}", "c3bf/d5f8 full90 Weave fallback"]}], ["expected_shape_wins", {"__tuple__": ["build_qm1024_d128_k8", "rag_online_b1_q1_m100000_d128_k10", "rag_online_b1_q1_m65536_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "rag_online_irregular_b1_q1_m262143_d128_k10", "rag_online_irregular_b1_q1_m524287_d128_k10", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "rag_microbatch_largek_b1_q16_m250000_d128_k32", "rag_microbatch_largek_b1_q24_m100000_d128_k32", "rag_stream_largek_b1_q128_m131071_d128_k32", "build_k_sweep_qm512_k10", "build_qm1024_d128_k10", "build_batch_b2_q1024_m1024_d128_k10", "build_tail_b1_q1536_m1536_d128_k10", "build_large_b1_q6144_m6144_d128_k10", "build_qm2048_d128_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "search_rect_b1_q1024_m8192_d128_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10"]}], ["fallback", "loom.examples.weave.knn_build_dispatch_1877_9a17_b644_6a35_full90_selective_v1:candidate_9a17_only_full90_v1"], ["rejected_reason", null]]}]}'))

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown fd37 FP16 full90 candidate ', format(repr(candidate_key), '')])) from exc

def _select_contract_shapes(shape_labels):
    return parent_full90._select_contract_shapes(shape_labels)

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    return parent_full90._trace_inputs_for_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return parent_full90._inputs_for_label(label)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    return parent_full90._normalize_route_row(row)

def _timing_backend_name(use_cupti: bool) -> str:
    return parent_full90._timing_backend_name(use_cupti)

def _payload_shape_labels(shape_labels) -> str | tuple[str, ...]:
    return parent_full90._payload_shape_labels(shape_labels)

def _denominator_name(shape_labels) -> str:
    return parent_full90._denominator_name(shape_labels)

def _denominator_label(shape_labels) -> str:
    return parent_full90._denominator_label(shape_labels)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return parent_full90._rows_for_labels(report, labels)

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return parent_full90._timing_backends_for_reports(*reports)

def _eligible_fp16_d128_k10(inputs: dict[str, Any]) -> bool:
    return fp16_fd37._eligible_fp16_s8_cached_merge(inputs)

def _expected_seed(inputs: dict[str, Any], candidate_key: str) -> str | None:
    if candidate_key == CANDIDATE_FP16 and _eligible_fp16_d128_k10(inputs):
        return SEED_FP16_D128_K10_ID
    return None

def _base_route(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    return parent_full90.route_for_contract_inputs(inputs, candidate_key=BASE_CANDIDATE_KEY, force_fallback=force_fallback)

def _base_launch(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    parent_full90.launch_from_contract_inputs(inputs, candidate_key=BASE_CANDIDATE_KEY, force_fallback=force_fallback)

def _base_trace_row(label: str, *, force_fallback: bool=False) -> dict[str, Any]:
    return dict(parent_full90.route_trace_for_contract_shapes((label,), candidate_key=BASE_CANDIDATE_KEY, force_fallback=force_fallback)[0])

def route_for_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> str:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_CANDIDATE_KEY:
        return _base_route(inputs, force_fallback=force_fallback)
    if _eligible_fp16_d128_k10(inputs):
        return ROUTE_FP16_D128_K10_SEED
    return _base_route(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> None:
    _candidate_config(candidate_key)
    if force_fallback or candidate_key == BASE_CANDIDATE_KEY:
        _base_launch(inputs, force_fallback=force_fallback)
        return
    if _eligible_fp16_d128_k10(inputs):
        fp16_fd37.launch_from_contract_inputs(inputs)
        return
    _base_launch(inputs)

def candidate_baseline_9a17_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=BASE_CANDIDATE_KEY)

def candidate_fp16_fd37_full90_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_FP16)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_fp16_fd37_full90_v1(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, candidate_key=CANDIDATE_FP16, force_fallback=True)

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], None]:
    if candidate_key == BASE_CANDIDATE_KEY:
        return candidate_baseline_9a17_full90_v1
    if candidate_key == CANDIDATE_FP16:
        return candidate_fp16_fd37_full90_v1
    raise ValueError(''.join(['unknown fd37 FP16 full90 candidate ', format(repr(candidate_key), '')]))

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def compile_and_launch_knn_build(*, shape_labels=None, benchmark: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    selected = None if shape_labels is None else _select_contract_shapes(shape_labels)
    report = evaluate_contract(shapes=selected, correctness=True, benchmark=benchmark, kernel_fn=_candidate_kernel_fn(candidate_key))
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=None, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return parent_full90._run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=kernel_fn, correctness=correctness, time_flashlib=time_flashlib)

def _fp16_trace_row(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    base_row = _base_trace_row(label, force_fallback=False)
    if force_fallback:
        row = _base_trace_row(label, force_fallback=True)
        row['expected_seed'] = SEED_FP16_D128_K10_ID
        row['guard_id'] = ''.join(['forced_fallback_', format(SEED_FP16_D128_K10_ID, ''), '_disabled'])
        row['guard_condition'] = ''.join(['forced fallback to 1877+9a17; ', format(SEED_FP16_D128_K10_ID, ''), ' disabled'])
        row['classification'] = 'guard-miss'
        row['parent_dispatcher_route'] = base_row.get('selected_route')
        row['baseline_dispatcher_route'] = base_row.get('selected_route')
        return _normalize_route_row(row)
    return _normalize_route_row({'shape_key': label, 'selected_route': ROUTE_FP16_D128_K10_SEED, 'selected_entrypoint': FP16_SEED_ENTRYPOINT, 'selected_seed': SEED_FP16_D128_K10_ID, 'expected_seed': SEED_FP16_D128_K10_ID, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': ''.join(['selective_fd37_', format(SEED_FP16_D128_K10_ID, '')]), 'guard_condition': 'exact FP16 build B=1 Q=M=2048 D=128 K=10 split8 cached merge', 'matched_label': TARGET_FP16_D128_K10, 'parent_dispatcher_route': base_row.get('selected_route'), 'parent_dispatcher_selected_seed': base_row.get('selected_seed'), 'baseline_dispatcher_route': base_row.get('selected_route'), 'targeted_seed_payload': TARGETED_SEED_PAYLOAD, 'coverage': 'full90 one-seed FP16 overlay before fd37-selected 1877+9a17 baseline', 'classification': 'unmeasured'})

def _route_trace_record(inputs: dict[str, Any], *, candidate_key: str, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    if candidate_key == BASE_CANDIDATE_KEY:
        return _normalize_route_row(_base_trace_row(label, force_fallback=force_fallback))
    if _eligible_fp16_d128_k10(inputs):
        return _fp16_trace_row(inputs, force_fallback=force_fallback)
    row = _base_trace_row(label, force_fallback=force_fallback)
    row['parent_dispatcher_route'] = _base_route(inputs, force_fallback=force_fallback)
    row['expected_seed'] = None
    return _normalize_route_row(row)

def route_trace_for_contract_shapes(shape_labels=None, *, candidate_key: str=DEFAULT_CANDIDATE_KEY, force_fallback: bool=False) -> list[dict[str, Any]]:
    _candidate_config(candidate_key)
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), candidate_key=candidate_key, force_fallback=force_fallback) for shape in selected]

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str, speedup_floor: float) -> list[dict[str, Any]]:
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
        out['baseline_9a17_kernel_ms'] = baseline_ms
        out['shape_specific_kernel_ms'] = candidate_ms if selected_new_seed else None
        out['external_baseline_ms'] = flashlib_ms
        out['flashlib_ms'] = flashlib_ms
        out['relative_speedup_vs_baseline'] = speedup_vs_baseline
        out['relative_speedup_vs_9a17'] = speedup_vs_baseline
        out['speedup_vs_external_baseline'] = speedup_vs_external
        out['external_baseline_ref'] = 'same_session' if flashlib_ms is not None else 'not_available'
        out['route_changed_vs_9a17'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        if expected_seed in new_seed_ids and selected_seed != expected_seed:
            out['classification'] = 'guard-miss'
        elif selected_new_seed and speedup_vs_external is not None and (speedup_vs_external >= speedup_floor):
            out['classification'] = 'seed-consumed'
        elif speedup_vs_external is not None and speedup_vs_external < speedup_floor:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
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

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = []
    for label in TARGET_SHAPES:
        inputs = _inputs_for_label(label)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        matrix.append({'shape_key': label, 'baseline_route': _base_route(inputs), 'candidate_route': route_for_contract_inputs(inputs), 'selected_seed': SEED_FP16_D128_K10_ID, 'candidate_id': CANDIDATE_ID, 'candidate_ms': candidate_ms, 'baseline_9a17_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'delta_ms_candidate_minus_9a17': candidate_ms - baseline_ms if candidate_ms is not None and baseline_ms is not None else None, 'speedup_vs_9a17': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_payload': TARGETED_SEED_PAYLOAD, 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def benchmark_baseline_9a17(*, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    return _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_9a17_full90_v1, correctness=benchmark_correctness, time_flashlib=time_flashlib)

def _baseline_sidecar(report: dict[str, Any], *, shape_labels, denominator: str, timing_backend: str, benchmark_correctness: bool, time_flashlib: bool, speedup_floor: float) -> dict[str, Any]:
    route_trace = route_trace_for_contract_shapes(shape_labels, candidate_key=BASE_CANDIDATE_KEY)
    below_1x = _below_flashlib_rows(report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(report, route_trace, floor=speedup_floor)
    return {'candidate_id': BASE_CANDIDATE_ID, 'candidate_key': BASE_CANDIDATE_KEY, 'selected_seeds': CANDIDATE_CONFIGS[BASE_CANDIDATE_KEY]['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': report['summary']['primary_mean'], 'all_correct': report['summary']['all_correct'], 'performance_comparable': report['summary']['performance_comparable'], 'invalid_performance_reason': report['summary']['invalid_performance_reason'], 'measured_entrypoint': BASE_BENCHMARK_ENTRYPOINT, 'route_entrypoint': BASE_ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'baseline_payload': None, 'speedup_floor': speedup_floor, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'route_trace': route_trace, 'route_trace_included': True, 'contract_summary': report['summary'], 'contract_performance': report['performance'], 'timing_backends': _timing_backends_for_reports(report), 'timing_backend_requested': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'report': report}

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, benchmark_correctness: bool, time_flashlib: bool, speedup_floor: float) -> dict[str, Any]:
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, candidate_key=CANDIDATE_FP16), candidate_report, baseline_report, candidate_key=CANDIDATE_FP16, speedup_floor=speedup_floor)
    below_1x = _below_flashlib_rows(candidate_report, route_trace, floor=1.0)
    below_floor = _below_flashlib_rows(candidate_report, route_trace, floor=speedup_floor)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    config = _candidate_config(CANDIDATE_FP16)
    metric_delta = candidate_metric - baseline_metric if candidate_metric is not None and baseline_metric is not None else None
    return {'candidate_id': config['candidate_id'], 'candidate_key': CANDIDATE_FP16, 'baseline_candidate_id': BASE_CANDIDATE_ID, 'baseline_candidate_key': BASE_CANDIDATE_KEY, 'selected_seeds': config['selected_seeds'], 'source_tasks': SOURCE_TASKS, 'tflops': candidate_metric, 'baseline_9a17_tflops': baseline_metric, 'metric_delta_vs_9a17': metric_delta, 'metric_delta_vs_baseline': metric_delta, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': BENCHMARK_ENTRYPOINT, 'baseline_entrypoint': BASE_BENCHMARK_ENTRYPOINT, 'route_entrypoint': ROUTE_ENTRYPOINT, 'measured_shape_labels': _payload_shape_labels(shape_labels), 'timing_backend': timing_backend, 'denominator': denominator, 'selected_route_labels': config['expected_shape_wins'], 'selected_route_rows': _rows_for_labels(candidate_report, TARGET_SHAPES), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, TARGET_SHAPES), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report), 'targeted_seed_payload': TARGETED_SEED_PAYLOAD, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': config['candidate_id'], 'guard_plan': config['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, candidate_key=CANDIDATE_FP16, force_fallback=True), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': timing_backend, 'use_cupti': use_cupti, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'performance_coverage': 'partial' if below_floor else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_floor, 'flashlib_parity_ledger': {'baseline_ref_scope': 'same_session' if time_flashlib else 'not_available', 'baseline_payload': None, 'speedup_floor': speedup_floor, 'rows_below_1x': below_1x, 'rows_below_floor': below_floor, 'omitted_reason': None if time_flashlib else 'benchmark_time_flashlib=false'}, 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_metric, 'baseline_9a17_value': baseline_metric, 'delta_vs_9a17': metric_delta, 'denominator': denominator}, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_fp16_fd37_full90_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None, benchmark_correctness: bool=True, time_flashlib: bool=True, speedup_floor: float=1.2) -> dict[str, Any]:
    if baseline_report is None:
        baseline_report = benchmark_baseline_9a17(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_fp16_fd37_full90_v1, correctness=benchmark_correctness, time_flashlib=time_flashlib)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib, speedup_floor=speedup_floor)

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, benchmark_correctness: bool=True, time_flashlib: bool=True, speedup_floor: float=1.2) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom_label = _denominator_label(shape_labels)
    denominator = _denominator_name(shape_labels)
    timing_backend = _timing_backend_name(use_cupti)
    baseline_report = benchmark_baseline_9a17(use_cupti=use_cupti, shape_labels=shape_labels, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib)
    baseline_payload = _baseline_sidecar(baseline_report, shape_labels=shape_labels, denominator=denominator, timing_backend=timing_backend, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib, speedup_floor=speedup_floor)
    candidate_payload = benchmark_candidate_fp16_fd37_full90_v1(use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report, benchmark_correctness=benchmark_correctness, time_flashlib=time_flashlib, speedup_floor=speedup_floor)
    baseline_path = out_dir / ''.join([format(denom_label, ''), '_same_session_baseline_9a17_v1.json'])
    payload_path = out_dir / ''.join([format(denom_label, ''), '_dispatch_1877_plus_9a17_fp16_fd37_v1.json'])
    trace_path = out_dir / ''.join([format(denom_label, ''), '_route_trace_1877_plus_9a17_fp16_fd37_v1.json'])
    forced_trace_path = out_dir / ''.join([format(denom_label, ''), '_forced_fallback_trace_1877_plus_9a17_fp16_fd37_v1.json'])
    seed_matrix_path = out_dir / ''.join([format(denom_label, ''), '_seed_delta_matrix_1877_plus_9a17_fp16_fd37_v1.json'])
    summary_path = out_dir / ''.join([format(denom_label, ''), '_dispatcher_consumption_1877_9a17_fp16_fd37_v1.json'])
    artifacts: dict[str, str] = {'same_session_baseline_payload': str(baseline_path), 'candidate_payload': str(payload_path), 'route_trace': str(trace_path), 'forced_fallback_trace': str(forced_trace_path), 'seed_delta_matrix': str(seed_matrix_path), 'dispatcher_consumption': str(summary_path)}
    baseline_payload['flashlib_parity_ledger']['baseline_payload'] = str(baseline_path)
    candidate_payload['flashlib_parity_ledger']['baseline_payload'] = str(baseline_path)
    summary = {'candidate_id': 'dispatcher_consumption_1877_9a17_fp16_fd37_full90_v1', 'measured_entrypoint': ''.join([format(MODULE, ''), ':write_benchmark_artifacts']), 'denominator': denominator, 'timing_backend': timing_backend, 'benchmark_correctness_checked': benchmark_correctness, 'benchmark_time_flashlib': time_flashlib, 'baseline_candidate_key': BASE_CANDIDATE_KEY, 'selected_candidate_key': CANDIDATE_FP16, 'selected_candidate_dispatcher': CANDIDATE_ID, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'seed_delta_matrix': candidate_payload['seed_delta_matrix'], 'flashlib_parity_ledger': candidate_payload['flashlib_parity_ledger'], 'full_denominator_ab': {'baseline_payload': str(baseline_path), 'candidate_payload': str(payload_path), 'denominator': denominator, 'timing_backend': timing_backend, 'metric_delta': candidate_payload.get('metric_delta_vs_9a17'), 'route_trace': candidate_payload.get('route_trace', [])}, 'baseline_tflops': baseline_payload.get('tflops'), 'candidate_tflops': candidate_payload.get('tflops'), 'metric_delta_vs_9a17': candidate_payload.get('metric_delta_vs_9a17'), 'metric_delta_vs_baseline': candidate_payload.get('metric_delta_vs_baseline'), 'artifacts': artifacts}
    baseline_path.write_text(json.dumps(baseline_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    payload_path.write_text(json.dumps(candidate_payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    trace_path.write_text(json.dumps(candidate_payload['route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    forced_trace_path.write_text(json.dumps(candidate_payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    seed_matrix_path.write_text(json.dumps(candidate_payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n', encoding='utf-8')
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return artifacts
