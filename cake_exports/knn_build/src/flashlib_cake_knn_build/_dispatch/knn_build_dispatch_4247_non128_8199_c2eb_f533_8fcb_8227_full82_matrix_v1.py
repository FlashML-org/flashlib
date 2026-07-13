"""Full82 split148 subset-matrix kNN build dispatcher over 8199 widecombine.

Minimum target architecture: sm_100a. This opt-in dispatcher candidate starts
from the 4247+8199-widecombine full82 Weave dispatcher and overlays exact
guards for the current additive seeds: c2eb D96 exact, 8227 D320 exact-tail,
f533 D768 fused-merge, and 8fcb split148 K32 mixed-route. Guard misses stay on the
8199-widecombine full82 Weave dispatcher; no external implementation is on the
production route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_4247_non128_8199_widecombine_full82_v1 as baseline_wide
from . import knn_build_dispatch_e3de_9138_bcb3_4247_v1 as base_4247
from . import knn_build_non128_frontier_4be7_d768fused_v1 as seed_d768
from . import knn_build_non128_frontier_4be7_d96exact_v1 as seed_d96
from . import knn_build_non128_frontier_8227_d320tail_v1 as seed_d320
from . import knn_build_rag_microbucket_k32_8fcb_split148_v1 as seed_k32
MODULE = 'loom.examples.weave.knn_build_dispatch_4247_non128_8199_c2eb_f533_8fcb_8227_full82_matrix_v1'
BASELINE_ID = 'baseline_4247_non128_8199_widecombine_full82_v1'
BASELINE_ENTRYPOINT = 'loom.examples.weave.knn_build_dispatch_4247_non128_8199_widecombine_full82_v1:benchmark_knn_build_dispatch_4247_non128_8199_widecombine_full82_v1'
SEED_D96_ID = 'non128_frontier_4be7_d96exact_v1'
SEED_D320_ID = 'non128_frontier_8227_d320tail_v1'
SEED_D768_ID = 'non128_frontier_4be7_d768fused_v1'
SEED_K32_ID = seed_k32.SEED_K32_8FCB_SPLIT148_ID
ROUTE_D96_ENTRYPOINT = ''.join([format(seed_d96.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_D320_ENTRYPOINT = ''.join([format(seed_d320.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_D768_ENTRYPOINT = ''.join([format(seed_d768.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_K32_ENTRYPOINT = ''.join([format(seed_k32.MODULE, ''), ':launch_from_contract_inputs'])
ROUTE_BASELINE_ENTRYPOINT = ''.join([format(baseline_wide.MODULE, ''), ':launch_from_contract_inputs'])
D96_TARGET_SHAPES = (seed_d96.D96_SHAPE,)
D320_TARGET_SHAPES = (seed_d320.D320_BUILD_SHAPE, seed_d320.D320_SEARCH_SHAPE)
D768_TARGET_SHAPES = (seed_d768.D768_SHAPE,)
K32_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32"]}'))
CONSUMED_SEED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d96_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32"]}'))
TARGET_SHAPES = CONSUMED_SEED_TARGET_SHAPES
TARGET_SHAPE_SET = set(TARGET_SHAPES)
SELECTED_TARGET_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "build_dim_sweep_b1_q1024_m1024_d96_k10", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32"]}'))
GUARD_MISS_AUDIT_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["build_dim_sweep_b1_q1024_m1024_d96_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_large_b1_q8192_m8192_d128_k20", "build_qm2048_d128_k10", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "flashml_correctness_b1_q256_m256_d128_k5", "build_over32_stress_qm2048_k64", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "build_k_sweep_qm512_k5", "build_over32_stress_qm4096_k64"]}'))
DISPATCH_CORRECTNESS_SHAPES = _decode_capture(_json_loads('{"__tuple__": ["flashml_correctness_b1_q256_m256_d128_k5", "build_dim_sweep_b1_q1024_m1024_d96_k10", "build_highd_b1_q1024_m1024_d320_k10", "search_rect_highd_b1_q512_m12000_d320_k10", "rag_microbatch_highd_b1_q16_m50000_d768_k10", "rag_microbatch_largek_b1_q8_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m100000_d128_k32", "rag_microbatch_largek_b1_q32_m100000_d128_k32", "rag_microbatch_largek_b1_q16_m131071_d128_k32", "build_dim_sweep_b1_q2048_m2048_d192_k10", "build_large_b1_q8192_m8192_d128_k20", "build_large_b1_q8192_m8192_d128_k32", "build_over64_stress_qm2048_k96", "rag_microbatch_b1_q16_m100000_d128_k10", "rag_stream_largek_b1_q128_m100000_d128_k32", "rag_batch_b2_q256_m50000_d128_k10", "rag_irregular_b1_q512_m131071_d128_k10", "search_rect_b1_q1024_m8192_d128_k10", "build_dim_sweep_b1_q2048_m2048_d64_k10", "build_dim_sweep_b1_q2048_m2048_d256_k10", "build_dtype_fp16_b1_q2048_m2048_d128_k10", "search_rect_b1_q2048_m32768_d128_k10", "build_large_tail_b1_q6144_m6144_d128_k20", "build_k_sweep_qm2048_k24", "build_k_sweep_qm2048_k28", "build_k_sweep_qm4096_k28", "build_over32_stress_qm2048_k64", "build_over32_stress_qm4096_k64", "build_k_sweep_qm512_k1", "build_k_sweep_qm512_k2", "build_k_sweep_qm1024_k16", "build_k_sweep_qm512_k4", "build_k_sweep_qm512_k5", "build_k_sweep_qm512_k6", "rag_microbatch_b1_q8_m100000_d128_k10", "rag_microbatch_b1_q32_m100000_d128_k10", "build_dim_sweep_b1_q1024_m1024_d64_k10", "build_dim_sweep_b1_q4096_m4096_d64_k10", "search_rect_b1_q1024_m32768_d64_k10", "rag_online_b1_q1_m100000_d128_k10", "rag_online_irregular_b1_q1_m131071_d128_k10", "rag_online_large_m_b1_q1_m250000_d128_k10", "build_over64_stress_qm1024_k96", "build_over64_stress_qm4096_k96", "build_k_sweep_qm512_k8", "build_k_sweep_qm512_k10", "build_qm2048_d128_k10", "search_rect_over32_b1_q2048_m65536_d128_k64", "rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", "rag_stream_b1_q128_m100000_d128_k10", "search_rect_b1_q4096_m65536_d128_k20", "rag_offline_largek_b1_q4096_m100000_d128_k20", "rag_offline_large_m_b1_q8192_m250000_d128_k20"]}'))
TARGETED_SEED_ROWS: dict[str, dict[str, Any]] = {seed_d96.D96_SHAPE: {'kernel_ms': 0.047168, 'flashlib_ms': 0.046848, 'ratio_vs_flashlib': 0.9932157394843962, 'tflops': 4.268287652645862}, seed_d320.D320_BUILD_SHAPE: {'kernel_ms': 0.047489, 'flashlib_ms': 0.070528, 'ratio_vs_flashlib': 1.4851439280675522, 'tflops': 14.131454442081324}, seed_d320.D320_SEARCH_SHAPE: {'kernel_ms': 0.1148, 'flashlib_ms': 0.151615, 'ratio_vs_flashlib': 1.3206881533101045, 'tflops': 34.25226480836237}, seed_d768.D768_SHAPE: {'kernel_ms': 0.094592, 'flashlib_ms': 0.167136, 'ratio_vs_flashlib': 1.7669147496617053, 'tflops': 12.990527740189446}, seed_k32.Q8_K32_SHAPE: {'kernel_ms': 0.077121, 'flashlib_ms': 0.107232, 'ratio_vs_flashlib': 1.3904384019916753, 'tflops': 2.655567225528715}, seed_k32.Q16_K32_SHAPE: {'kernel_ms': 0.135552, 'flashlib_ms': 0.134017, 'ratio_vs_flashlib': 0.988675932483475, 'tflops': 3.021718602455146}, seed_k32.Q16_K32_IRREGULAR_SHAPE: {'kernel_ms': 0.163168, 'flashlib_ms': 0.157824, 'ratio_vs_flashlib': 0.967248480094136, 'tflops': 3.2902702490684446}, seed_k32.Q32_K32_SHAPE: {'kernel_ms': 0.141312, 'flashlib_ms': 0.159264, 'ratio_vs_flashlib': 1.1270380434782608, 'tflops': 5.797101449275363}}
_CONTRACT_PARAMS_BY_LABEL = _decode_capture(_json_loads('{"__dict_items__": [["flashml_correctness_b1_q256_m256_d128_k5", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 256], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 606001], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.99], ["benchmark", true], ["time_flashlib", true]]}], ["build_k_sweep_qm512_k1", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 606049], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_k_sweep_qm512_k2", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 128], ["K", 2], ["dtype", "bfloat16"], ["seed", 606050], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_k_sweep_qm512_k4", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 128], ["K", 4], ["dtype", "bfloat16"], ["seed", 606052], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "low_k_q512_k5_neighborhood"]]}], ["build_k_sweep_qm512_k5", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 606053], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_k_sweep_qm512_k6", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 128], ["K", 6], ["dtype", "bfloat16"], ["seed", 606054], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "low_k_q512_k5_neighborhood"]]}], ["build_k_sweep_qm512_k8", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 606056], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_k_sweep_qm512_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606058], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_qm1024_d128_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606104], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_k_sweep_qm1024_k16", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 128], ["K", 16], ["dtype", "bfloat16"], ["seed", 606116], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "mid_k_topk_bucket"]]}], ["build_k_sweep_qm1024_k12", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 128], ["K", 12], ["dtype", "bfloat16"], ["seed", 606112], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_k_sweep_qm1024_k20", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 606120], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_qm2048_d128_k8", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 606208], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_qm1024_d128_k8", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 611108], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "low_k_build_frontier"]]}], ["build_qm4096_d128_k8", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 611408], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "low_k_build_frontier"]]}], ["build_qm2048_d128_k10", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606210], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_dim_sweep_b1_q1024_m1024_d64_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 64], ["K", 10], ["dtype", "bfloat16"], ["seed", 608164], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "d64_dispatch_guard_blindspot"]]}], ["build_dim_sweep_b1_q2048_m2048_d64_k10", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 64], ["K", 10], ["dtype", "bfloat16"], ["seed", 606264], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "d_generalization"]]}], ["build_dim_sweep_b1_q4096_m4096_d64_k10", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 64], ["K", 10], ["dtype", "bfloat16"], ["seed", 608464], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "d64_dispatch_guard_blindspot"]]}], ["build_dim_sweep_b1_q1024_m1024_d96_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 96], ["K", 10], ["dtype", "bfloat16"], ["seed", 610096], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "d_generalization_non128"]]}], ["build_dim_sweep_b1_q2048_m2048_d192_k10", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 192], ["K", 10], ["dtype", "bfloat16"], ["seed", 610192], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "d_generalization_non128"]]}], ["build_dim_sweep_b1_q2048_m2048_d256_k10", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 606256], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "d_generalization"]]}], ["build_common_d256_b1_q1024_m1024_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 612256], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "common_embedding_dim_frontier"]]}], ["build_common_d768_b1_q1024_m1024_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 612768], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "common_embedding_dim_frontier"]]}], ["build_common_d1024_b1_q512_m512_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 1024], ["K", 10], ["dtype", "bfloat16"], ["seed", 613024], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "common_embedding_dim_frontier"]]}], ["build_common_d4096_b1_q512_m512_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 512], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 614096], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "common_embedding_dim_frontier"]]}], ["build_highd_b1_q1024_m1024_d320_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 320], ["K", 10], ["dtype", "bfloat16"], ["seed", 610320], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "high_d_split_generalization"]]}], ["build_dtype_fp16_b1_q2048_m2048_d128_k10", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 10], ["dtype", "float16"], ["seed", 606216], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "dtype_generalization"]]}], ["build_batch_b2_q1024_m1024_d128_k10", {"__dict_items__": [["B", 2], ["Q", 1024], ["M", 1024], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606211], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "batch_generalization"]]}], ["build_k_sweep_qm2048_k11", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 11], ["dtype", "bfloat16"], ["seed", 609211], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "mid_k_boundary_guard"]]}], ["build_k_sweep_qm2048_k12", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 12], ["dtype", "bfloat16"], ["seed", 606212], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "mid_k_boundary_guard"]]}], ["build_k_sweep_qm2048_k13", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 13], ["dtype", "bfloat16"], ["seed", 609213], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "mid_k_boundary_guard"]]}], ["build_k_sweep_qm2048_k20", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 606220], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_k_sweep_qm2048_k24", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 24], ["dtype", "bfloat16"], ["seed", 606224], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "large_k_topk_ramp"]]}], ["build_k_sweep_qm2048_k28", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 28], ["dtype", "bfloat16"], ["seed", 606228], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "large_k_topk_ramp"]]}], ["build_tail_b1_q1536_m1536_d128_k10", {"__dict_items__": [["B", 1], ["Q", 1536], ["M", 1536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606153], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "non_power_of_two_build"]]}], ["build_tail_b1_q3072_m3072_d128_k20", {"__dict_items__": [["B", 1], ["Q", 3072], ["M", 3072], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 606327], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "non_power_of_two_mid_k_build"]]}], ["build_medium_b1_q4096_m4096_d128_k10", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606002], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_k_sweep_qm4096_k12", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 12], ["dtype", "bfloat16"], ["seed", 606412], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "mid_k_boundary_guard"]]}], ["build_k_sweep_qm4096_k13", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 13], ["dtype", "bfloat16"], ["seed", 609413], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "mid_k_boundary_guard"]]}], ["build_k_sweep_qm4096_k20", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 606420], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_k_sweep_qm4096_k24", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 24], ["dtype", "bfloat16"], ["seed", 606424], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "large_k_topk_ramp"]]}], ["build_k_sweep_qm4096_k28", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 28], ["dtype", "bfloat16"], ["seed", 606428], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "large_k_topk_ramp"]]}], ["build_largek_stress_qm4096_k32", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 606432], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "large_k_topk_bottleneck"]]}], ["build_k_sweep_qm4096_k30", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 30], ["dtype", "bfloat16"], ["seed", 606430], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "large_k_topk_bottleneck"]]}], ["build_over32_stress_qm2048_k48", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 48], ["dtype", "bfloat16"], ["seed", 606248], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "over32_topk_bottleneck"]]}], ["build_over32_stress_qm2048_k64", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 606249], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "over32_topk_bottleneck"]]}], ["build_over32_stress_qm4096_k48", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 48], ["dtype", "bfloat16"], ["seed", 606448], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "over32_topk_bottleneck"]]}], ["build_large_b1_q8192_m8192_d128_k10", {"__dict_items__": [["B", 1], ["Q", 8192], ["M", 8192], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606819], ["build", true], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["build_large_b1_q6144_m6144_d128_k10", {"__dict_items__": [["B", 1], ["Q", 6144], ["M", 6144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611614], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "large_build_k10_interpolation"]]}], ["build_large_b1_q8192_m8192_d128_k20", {"__dict_items__": [["B", 1], ["Q", 8192], ["M", 8192], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 606820], ["build", true], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "large_build_mid_k"]]}], ["build_large_b1_q8192_m8192_d128_k32", {"__dict_items__": [["B", 1], ["Q", 8192], ["M", 8192], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 606832], ["build", true], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "large_build_large_k"]]}], ["build_verylarge_b1_q12288_m12288_d128_k10", {"__dict_items__": [["B", 1], ["Q", 12288], ["M", 12288], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606123], ["build", true], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "very_large_build"]]}], ["rag_offline_b1_q4096_m100000_d128_k10", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 100000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606902], ["build", false], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["search_rect_b1_q1024_m8192_d128_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 8192], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606812], ["build", false], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rectangular_search"]]}], ["search_rect_b1_q1024_m32768_d64_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 32768], ["D", 64], ["K", 10], ["dtype", "bfloat16"], ["seed", 608864], ["build", false], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rectangular_search_d64_guard_blindspot"]]}], ["search_rect_highd_b1_q512_m12000_d320_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 12000], ["D", 320], ["K", 10], ["dtype", "bfloat16"], ["seed", 610321], ["build", false], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "flashlib_high_d_search"]]}], ["search_rect_common_d256_b1_q1024_m32768_k10", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 32768], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 612856], ["build", false], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "common_embedding_dim_rectangular_search"]]}], ["search_rect_common_d768_b1_q512_m8192_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 8192], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 612876], ["build", false], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "common_embedding_dim_rectangular_search"]]}], ["search_rect_b1_q4096_m65536_d128_k20", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 65536], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 606655], ["build", false], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rectangular_search_mid_k"]]}], ["search_rect_b1_q1536_m65536_d128_k20", {"__dict_items__": [["B", 1], ["Q", 1536], ["M", 65536], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 608655], ["build", false], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rectangular_search_tail_guard"]]}], ["search_rect_over32_b1_q2048_m65536_d128_k64", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 65536], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 606664], ["build", false], ["check_correctness", true], ["correctness_query_sample", 64], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rectangular_search_over32"]]}], ["rag_online_b1_q1_m100000_d128_k10", {"__dict_items__": [["B", 1], ["Q", 1], ["M", 100000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606901], ["build", false], ["check_correctness", true], ["correctness_query_sample", 1], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["public_baseline_qps", 2600], ["diagnostic_class", "rag_online_single"]]}], ["rag_online_b1_q1_m65536_d128_k10", {"__dict_items__": [["B", 1], ["Q", 1], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611901], ["build", false], ["check_correctness", true], ["correctness_query_sample", 1], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["public_baseline_qps", 2600], ["diagnostic_class", "rag_online_single_low_m_frontier"]]}], ["rag_online_irregular_b1_q1_m131071_d128_k10", {"__dict_items__": [["B", 1], ["Q", 1], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 608191], ["build", false], ["check_correctness", true], ["correctness_query_sample", 1], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_online_single_m_guard_blindspot"]]}], ["rag_online_large_m_b1_q1_m250000_d128_k10", {"__dict_items__": [["B", 1], ["Q", 1], ["M", 250000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 608194], ["build", false], ["check_correctness", true], ["correctness_query_sample", 1], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_online_single_m_guard_blindspot"]]}], ["rag_online_irregular_b1_q1_m262143_d128_k10", {"__dict_items__": [["B", 1], ["Q", 1], ["M", 262143], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 608195], ["build", false], ["check_correctness", true], ["correctness_query_sample", 1], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_online_single_m_guard_blindspot"]]}], ["rag_online_irregular_b1_q1_m524287_d128_k10", {"__dict_items__": [["B", 1], ["Q", 1], ["M", 524287], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611905], ["build", false], ["check_correctness", true], ["correctness_query_sample", 1], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_online_single_large_m_frontier"]]}], ["rag_stream_b1_q128_m100000_d128_k10", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 100000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606928], ["build", false], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["public_baseline_qps", 5000], ["diagnostic_class", "rag_streaming"]]}], ["rag_offline_largek_b1_q4096_m100000_d128_k20", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 100000], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 606920], ["build", false], ["check_correctness", true], ["correctness_query_sample", 64], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_large_k_topk"]]}], ["rag_offline_large_m_b1_q8192_m250000_d128_k20", {"__dict_items__": [["B", 1], ["Q", 8192], ["M", 250000], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 606925], ["build", false], ["check_correctness", true], ["correctness_query_sample", 64], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_large_m_mid_k"]]}], ["rag_offline_large_m_over32_b1_q2048_m250000_d128_k64", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 250000], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 606964], ["build", false], ["check_correctness", true], ["correctness_query_sample", 64], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_large_m_over32"]]}], ["rag_offline_batch_b1_q10000_m100000_d128_k10", {"__dict_items__": [["B", 1], ["Q", 10000], ["M", 100000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606003], ["build", false], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["public_baseline_qps", 310000]]}], ["rag_offline_b1_q10000_m50000_d128_k10", {"__dict_items__": [["B", 1], ["Q", 10000], ["M", 50000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 606904], ["build", false], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true]]}], ["rag_microbatch_b1_q4_m100000_d128_k10", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 100000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 608004], ["build", false], ["check_correctness", true], ["correctness_query_sample", 4], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_microbatch_q_guard_blindspot"]]}], ["rag_microbatch_b1_q8_m100000_d128_k10", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 100000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 607008], ["build", false], ["check_correctness", true], ["correctness_query_sample", 8], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_microbatch_guard_gap"]]}], ["rag_microbatch_b1_q16_m100000_d128_k10", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 100000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 607016], ["build", false], ["check_correctness", true], ["correctness_query_sample", 16], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_microbatch_guard_gap"]]}], ["rag_microbatch_highd_b1_q16_m50000_d768_k10", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 50000], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 610768], ["build", false], ["check_correctness", true], ["correctness_query_sample", 16], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_embedding_dim_generalization"]]}], ["rag_microbatch_common_d64_b1_q16_m50000_k10", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 50000], ["D", 64], ["K", 10], ["dtype", "bfloat16"], ["seed", 612064], ["build", false], ["check_correctness", true], ["correctness_query_sample", 16], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "common_embedding_dim_rag_microbatch"]]}], ["rag_microbatch_common_d256_b1_q16_m50000_k10", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 50000], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 612266], ["build", false], ["check_correctness", true], ["correctness_query_sample", 16], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "common_embedding_dim_rag_microbatch"]]}], ["rag_microbatch_common_d1024_b1_q8_m50000_k10", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 50000], ["D", 1024], ["K", 10], ["dtype", "bfloat16"], ["seed", 613034], ["build", false], ["check_correctness", true], ["correctness_query_sample", 8], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "common_embedding_dim_rag_microbatch"]]}], ["rag_microbatch_common_d4096_b1_q4_m32768_k10", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 32768], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 614106], ["build", false], ["check_correctness", true], ["correctness_query_sample", 4], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "common_embedding_dim_rag_microbatch"]]}], ["rag_microbatch_b1_q32_m100000_d128_k10", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 100000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 607032], ["build", false], ["check_correctness", true], ["correctness_query_sample", 32], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_microbatch_guard_gap"]]}], ["rag_microbatch_largek_b1_q8_m100000_d128_k32", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 100000], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 608308], ["build", false], ["check_correctness", true], ["correctness_query_sample", 8], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_microbatch_large_k_guard_blindspot"]]}], ["rag_microbatch_largek_b1_q16_m100000_d128_k32", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 100000], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 608316], ["build", false], ["check_correctness", true], ["correctness_query_sample", 16], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_microbatch_large_k_guard_blindspot"]]}], ["rag_microbatch_largek_b1_q24_m100000_d128_k32", {"__dict_items__": [["B", 1], ["Q", 24], ["M", 100000], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 611324], ["build", false], ["check_correctness", true], ["correctness_query_sample", 24], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_microbatch_large_k_q_interpolation"]]}], ["rag_microbatch_largek_b1_q16_m250000_d128_k32", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 250000], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 611325], ["build", false], ["check_correctness", true], ["correctness_query_sample", 16], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_microbatch_large_k_large_m_frontier"]]}], ["rag_microbatch_largek_b1_q32_m100000_d128_k32", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 100000], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 608332], ["build", false], ["check_correctness", true], ["correctness_query_sample", 32], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_microbatch_large_k_guard_blindspot"]]}], ["rag_microbatch_largek_b1_q16_m131071_d128_k32", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 131071], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 608317], ["build", false], ["check_correctness", true], ["correctness_query_sample", 16], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_microbatch_large_k_irregular_m_guard"]]}], ["rag_microbatch_b1_q64_m100000_d128_k10", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 100000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 608064], ["build", false], ["check_correctness", true], ["correctness_query_sample", 64], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_microbatch_q_guard_blindspot"]]}], ["rag_stream_largek_b1_q128_m100000_d128_k32", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 100000], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 607128], ["build", false], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_streaming_large_k"]]}], ["rag_stream_largek_b1_q128_m131071_d128_k32", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131071], ["D", 128], ["K", 32], ["dtype", "bfloat16"], ["seed", 611328], ["build", false], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_streaming_large_k_irregular_m_frontier"]]}], ["rag_batch_b2_q256_m50000_d128_k10", {"__dict_items__": [["B", 2], ["Q", 256], ["M", 50000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 607256], ["build", false], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_batch_nonbuild"]]}], ["rag_irregular_b1_q512_m131071_d128_k10", {"__dict_items__": [["B", 1], ["Q", 512], ["M", 131071], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 607512], ["build", false], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rag_irregular_m_tail"]]}], ["search_rect_b1_q2048_m32768_d128_k10", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 607048], ["build", false], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "rectangular_search_intermediate"]]}], ["build_large_tail_b1_q6144_m6144_d128_k20", {"__dict_items__": [["B", 1], ["Q", 6144], ["M", 6144], ["D", 128], ["K", 20], ["dtype", "bfloat16"], ["seed", 607144], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "large_build_mid_k_interpolation"]]}], ["build_over32_stress_qm4096_k64", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 607464], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "over32_topk_bottleneck"]]}], ["build_over64_stress_qm1024_k96", {"__dict_items__": [["B", 1], ["Q", 1024], ["M", 1024], ["D", 128], ["K", 96], ["dtype", "bfloat16"], ["seed", 609196], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "over64_topk_bottleneck"]]}], ["build_over64_stress_qm2048_k96", {"__dict_items__": [["B", 1], ["Q", 2048], ["M", 2048], ["D", 128], ["K", 96], ["dtype", "bfloat16"], ["seed", 607296], ["build", true], ["check_correctness", true], ["correctness_query_sample", 512], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "over64_topk_bottleneck"]]}], ["build_over64_stress_qm4096_k96", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 4096], ["D", 128], ["K", 96], ["dtype", "bfloat16"], ["seed", 609496], ["build", true], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "over64_topk_bottleneck"]]}], ["rag_online_common_d64_b1_q1_m262143_k10", {"__dict_items__": [["B", 1], ["Q", 1], ["M", 262143], ["D", 64], ["K", 10], ["dtype", "bfloat16"], ["seed", 615064], ["build", false], ["check_correctness", true], ["correctness_query_sample", 1], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_common_d_rag_online_irregular"]]}], ["rag_microbatch_common_d64_b1_q4_m100000_k10", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 100000], ["D", 64], ["K", 10], ["dtype", "bfloat16"], ["seed", 615164], ["build", false], ["check_correctness", true], ["correctness_query_sample", 4], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_common_d_rag_microbatch_tail"]]}], ["rag_microbatch_common_d256_b1_q4_m100000_k10", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 100000], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 615256], ["build", false], ["check_correctness", true], ["correctness_query_sample", 4], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_common_d_rag_microbatch_tail"]]}], ["rag_stream_common_d256_b1_q128_m100000_k10", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 100000], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 615356], ["build", false], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_common_d_rag_streaming"]]}], ["rag_microbatch_common_d768_b1_q8_m100000_k10", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 100000], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 615768], ["build", false], ["check_correctness", true], ["correctness_query_sample", 8], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_common_d_rag_microbatch_tail"]]}], ["rag_microbatch_common_d1024_b1_q4_m100000_k10", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 100000], ["D", 1024], ["K", 10], ["dtype", "bfloat16"], ["seed", 616024], ["build", false], ["check_correctness", true], ["correctness_query_sample", 4], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_common_d_rag_microbatch_tail"]]}], ["rag_online_common_d4096_b1_q1_m65536_k10", {"__dict_items__": [["B", 1], ["Q", 1], ["M", 65536], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 616096], ["build", false], ["check_correctness", true], ["correctness_query_sample", 1], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_common_d_rag_online_highd"]]}], ["search_rect_common_d1024_b1_q256_m8192_k10", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 8192], ["D", 1024], ["K", 10], ["dtype", "bfloat16"], ["seed", 616124], ["build", false], ["check_correctness", true], ["correctness_query_sample", 256], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_common_d_rectangular_search"]]}], ["search_rect_common_d4096_b1_q128_m4096_k10", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 4096], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 616496], ["build", false], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_common_d_rectangular_search"]]}], ["rag_microbatch_largek_common_d256_b1_q8_m100000_k32", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 100000], ["D", 256], ["K", 32], ["dtype", "bfloat16"], ["seed", 616332], ["build", false], ["check_correctness", true], ["correctness_query_sample", 8], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_common_d_large_k_rag"]]}], ["rag_stream_largek_common_d256_b1_q128_m100000_k32", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 100000], ["D", 256], ["K", 32], ["dtype", "bfloat16"], ["seed", 616432], ["build", false], ["check_correctness", true], ["correctness_query_sample", 128], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_common_d_large_k_rag"]]}], ["rag_microbatch_over32_d128_b1_q16_m100000_k48", {"__dict_items__": [["B", 1], ["Q", 16], ["M", 100000], ["D", 128], ["K", 48], ["dtype", "bfloat16"], ["seed", 616548], ["build", false], ["check_correctness", true], ["correctness_query_sample", 16], ["recall_min", 0.999], ["benchmark", true], ["time_flashlib", true], ["diagnostic_class", "v12_rag_over32_topk"]]}]]}'))
_K32_SHAPE_SPECS = {label: _CONTRACT_PARAMS_BY_LABEL[label] for label in K32_TARGET_SHAPES}
PRODUCTION_ROUTE_MODULES = {**baseline_wide.PRODUCTION_ROUTE_MODULES, SEED_D96_ID: ROUTE_D96_ENTRYPOINT, SEED_D320_ID: ROUTE_D320_ENTRYPOINT, SEED_D768_ID: ROUTE_D768_ENTRYPOINT, SEED_K32_ID: ROUTE_K32_ENTRYPOINT, BASELINE_ID: ROUTE_BASELINE_ENTRYPOINT}
CANDIDATE_DISPATCHERS = ({'id': BASELINE_ID, 'entrypoint': BASELINE_ENTRYPOINT, 'consumed_seeds': (baseline_wide.SEED_NON128_ID,), 'guard_plan': ('8199 widecombine exact non-D128 guards', 'then exported 4247 guard stack'), 'expected_shape_wins': baseline_wide.TARGET_SHAPES, 'fallback': baseline_wide.ROUTE_BASE_4247_ENTRYPOINT, 'rejected_reason': 'same-session full82 baseline'}, {'id': 'candidate_d96_d320_only_over_8199_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_d96_d320_only']), 'consumed_seeds': (SEED_D96_ID, SEED_D320_ID), 'guard_plan': ('c2eb exact D96 guard', '8227 exact D320-tail guards', 'then 8199-widecombine full82 baseline'), 'expected_shape_wins': (*D96_TARGET_SHAPES, *D320_TARGET_SHAPES), 'fallback': ROUTE_BASELINE_ENTRYPOINT, 'rejected_reason': 'matrix candidate; promotion depends on same-session full82 A/B'}, {'id': 'candidate_f533_non128_over_8199_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_f533_non128_only']), 'consumed_seeds': (SEED_D96_ID, SEED_D320_ID, SEED_D768_ID), 'guard_plan': ('c2eb exact D96 guard', '8227 exact D320-tail guards', 'f533 exact D768 fused-merge guard', 'then 8199-widecombine full82 baseline'), 'expected_shape_wins': (*D96_TARGET_SHAPES, *D320_TARGET_SHAPES, *D768_TARGET_SHAPES), 'fallback': ROUTE_BASELINE_ENTRYPOINT, 'rejected_reason': 'matrix candidate; promotion depends on same-session full82 A/B'}, {'id': 'candidate_k32_split148_over_8199_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_k32_only']), 'consumed_seeds': (SEED_K32_ID,), 'guard_plan': ('8fcb split148 exact K32 RAG microbucket guards', 'then 8199-widecombine full82 baseline'), 'expected_shape_wins': K32_TARGET_SHAPES, 'fallback': ROUTE_BASELINE_ENTRYPOINT, 'rejected_reason': 'matrix candidate; promotion depends on same-session full82 A/B'}, {'id': 'candidate_f533_split148_over_8199_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_4247_non128_8199_c2eb_f533_8fcb_8227_full82_matrix_v1']), 'consumed_seeds': (SEED_D96_ID, SEED_D320_ID, SEED_D768_ID, SEED_K32_ID), 'guard_plan': ('c2eb exact D96 guard', '8227 exact D320-tail guards', 'f533 exact D768 fused-merge guard', '8fcb split148 exact K32 RAG microbucket guards', 'then 8199-widecombine full82 baseline'), 'expected_shape_wins': TARGET_SHAPES, 'fallback': ROUTE_BASELINE_ENTRYPOINT, 'rejected_reason': 'matrix candidate; promotion depends on same-session full82 A/B'})
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_evolve_7bfc_split_cg2_u2_stage1_k32_unordered_k48over32", "arg_keys": ["tmap_query", "tmap_database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tile_pairs", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [2, 1, 1], "computed_smem_bytes": 50432, "constants": [["BLOCK_Q", 128], ["BLOCK_M", 64], ["FEAT_D", 128], ["TOP_K_MAX", 48]], "cta_group": 1, "threads": 192}'))

def _dtype_name(inputs: dict[str, Any]) -> str:
    dtype = inputs.get('dtype')
    if dtype is not None:
        return str(dtype).replace('torch.', '')
    query = inputs.get('query')
    if query is not None and hasattr(query, 'dtype'):
        return str(query.dtype).replace('torch.', '')
    return ''

def _matches_contract_spec(inputs: dict[str, Any], spec: dict[str, Any]) -> bool:
    return int(inputs.get('B', -1)) == int(spec['B']) and int(inputs.get('Q', -1)) == int(spec['Q']) and (int(inputs.get('M', -1)) == int(spec['M'])) and (int(inputs.get('D', -1)) == int(spec['D'])) and (int(inputs.get('K', -1)) == int(spec['K'])) and (bool(inputs.get('build', False)) == bool(spec.get('build', False))) and (_dtype_name(inputs) == str(spec.get('dtype', 'bfloat16')).replace('torch.', ''))

def _d96_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = seed_d96._target_label_for_inputs(inputs)
    if label == seed_d96.D96_SHAPE:
        return label
    return None

def _d320_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = seed_d320._target_label_for_inputs(inputs)
    if label in D320_TARGET_SHAPES:
        return label
    return None

def _d768_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = seed_d768._target_label_for_inputs(inputs)
    if label == seed_d768.D768_SHAPE:
        return label
    return None

def _k32_label_for_inputs(inputs: dict[str, Any]) -> str | None:
    label = inputs.get('label')
    if label is not None:
        spec = _K32_SHAPE_SPECS.get(str(label))
        if spec is not None and _matches_contract_spec(inputs, spec):
            return str(label)
    for candidate_label, spec in _K32_SHAPE_SPECS.items():
        if _matches_contract_spec(inputs, spec):
            return candidate_label
    return None

def _selected_seed_for_inputs(inputs: dict[str, Any], *, enable_d96: bool=True, enable_d320tail: bool=True, enable_d768: bool=True, enable_k32: bool=True) -> tuple[str | None, str | None]:
    if enable_d96:
        label = _d96_label_for_inputs(inputs)
        if label is not None:
            return (SEED_D96_ID, label)
    if enable_d320tail:
        label = _d320_label_for_inputs(inputs)
        if label is not None:
            return (SEED_D320_ID, label)
    if enable_d768:
        label = _d768_label_for_inputs(inputs)
        if label is not None:
            return (SEED_D768_ID, label)
    if enable_k32:
        label = _k32_label_for_inputs(inputs)
        if label is not None:
            return (SEED_K32_ID, label)
    return (None, None)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_d96: bool=True, enable_d320tail: bool=True, enable_d768: bool=True, enable_k32: bool=True) -> str:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs, enable_d96=enable_d96, enable_d320tail=enable_d320tail, enable_d768=enable_d768, enable_k32=enable_k32)
        if selected_seed == SEED_D96_ID:
            return seed_d96.route_for_contract_inputs(inputs)
        if selected_seed == SEED_D320_ID:
            return seed_d320.route_for_contract_inputs(inputs)
        if selected_seed == SEED_D768_ID:
            return seed_d768.route_for_contract_inputs(inputs)
        if selected_seed == SEED_K32_ID:
            return seed_k32.route_for_contract_inputs(inputs)
    return baseline_wide.route_for_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False, enable_d96: bool=True, enable_d320tail: bool=True, enable_d768: bool=True, enable_k32: bool=True) -> None:
    if not force_fallback:
        selected_seed, _label = _selected_seed_for_inputs(inputs, enable_d96=enable_d96, enable_d320tail=enable_d320tail, enable_d768=enable_d768, enable_k32=enable_k32)
        if selected_seed == SEED_D96_ID:
            seed_d96.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_D320_ID:
            seed_d320.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_D768_ID:
            seed_d768.launch_from_contract_inputs(inputs)
            return
        if selected_seed == SEED_K32_ID:
            seed_k32.launch_from_contract_inputs(inputs)
            return
    baseline_wide.launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_f533_plus_k32(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate_f533_non128_only(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_k32=False)

def candidate_non128_only(inputs: dict[str, Any]) -> None:
    candidate_f533_non128_only(inputs)

def candidate_d96_d320_only(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_d768=False, enable_k32=False)

def candidate_k32_only(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, enable_d96=False, enable_d320tail=False, enable_d768=False)

def candidate_baseline_wide(inputs: dict[str, Any]) -> None:
    baseline_wide.launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)
CANDIDATE_KEYS = ('d96_d320_only', 'f533_non128_only', 'k32_only', 'f533_plus_k32')
DEFAULT_CANDIDATE_KEY = 'f533_plus_k32'
CANDIDATE_CONFIGS: dict[str, dict[str, Any]] = {'d96_d320_only': {'candidate_id': 'candidate_d96_d320_only_over_8199_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_d96_d320_only']), 'benchmark_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_d96_d320_only']), 'kernel_fn': candidate_d96_d320_only, 'enabled': {'enable_d96': True, 'enable_d320tail': True, 'enable_d768': False, 'enable_k32': False}, 'selected_seeds': (SEED_D96_ID, SEED_D320_ID), 'target_shapes': (*D96_TARGET_SHAPES, *D320_TARGET_SHAPES), 'guard_plan': ('c2eb exact D96 guard', '8227 exact D320-tail guards', '8199-widecombine fallback')}, 'f533_non128_only': {'candidate_id': 'candidate_f533_non128_over_8199_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_f533_non128_only']), 'benchmark_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_f533_non128_only']), 'kernel_fn': candidate_f533_non128_only, 'enabled': {'enable_d96': True, 'enable_d320tail': True, 'enable_d768': True, 'enable_k32': False}, 'selected_seeds': (SEED_D96_ID, SEED_D320_ID, SEED_D768_ID), 'target_shapes': (*D96_TARGET_SHAPES, *D320_TARGET_SHAPES, *D768_TARGET_SHAPES), 'guard_plan': ('c2eb exact D96 guard', '8227 exact D320-tail guards', 'f533 exact D768 fused-merge guard', '8199-widecombine fallback')}, 'k32_only': {'candidate_id': 'candidate_k32_split148_over_8199_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_k32_only']), 'benchmark_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_k32_only']), 'kernel_fn': candidate_k32_only, 'enabled': {'enable_d96': False, 'enable_d320tail': False, 'enable_d768': False, 'enable_k32': True}, 'selected_seeds': (SEED_K32_ID,), 'target_shapes': K32_TARGET_SHAPES, 'guard_plan': ('8fcb split148 exact K32 RAG microbucket guards', '8199-widecombine fallback')}, 'f533_plus_k32': {'candidate_id': 'candidate_f533_split148_over_8199_full82_v1', 'entrypoint': ''.join([format(MODULE, ''), ':candidate_f533_plus_k32']), 'benchmark_entrypoint': ''.join([format(MODULE, ''), ':benchmark_knn_build_dispatch_4247_non128_8199_c2eb_f533_8fcb_8227_full82_matrix_v1']), 'kernel_fn': candidate_f533_plus_k32, 'enabled': {'enable_d96': True, 'enable_d320tail': True, 'enable_d768': True, 'enable_k32': True}, 'selected_seeds': (SEED_D96_ID, SEED_D320_ID, SEED_D768_ID, SEED_K32_ID), 'target_shapes': TARGET_SHAPES, 'guard_plan': ('c2eb exact D96 guard', '8227 exact D320-tail guards', 'f533 exact D768 fused-merge guard', '8fcb split148 exact K32 RAG microbucket guards', '8199-widecombine fallback')}}

def _candidate_config(candidate_key: str) -> dict[str, Any]:
    try:
        return CANDIDATE_CONFIGS[candidate_key]
    except KeyError as exc:
        raise ValueError(''.join(['unknown candidate key ', format(repr(candidate_key), ''), '; expected one of ', format(CANDIDATE_KEYS, '')])) from exc

def _candidate_enabled_kwargs(candidate_key: str) -> dict[str, bool]:
    return dict(_candidate_config(candidate_key)['enabled'])

def _candidate_target_shapes(candidate_key: str) -> tuple[str, ...]:
    return tuple(_candidate_config(candidate_key)['target_shapes'])

def _candidate_selected_seeds(candidate_key: str) -> tuple[str, ...]:
    return tuple(_candidate_config(candidate_key)['selected_seeds'])

def _candidate_kernel_fn(candidate_key: str) -> Callable[[dict[str, Any]], Any]:
    return _candidate_config(candidate_key)['kernel_fn']

def _candidate_id(candidate_key: str) -> str:
    return str(_candidate_config(candidate_key)['candidate_id'])

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return baseline_wide._select_contract_shapes(shape_labels)

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
    return baseline_wide._trace_inputs_from_shape(shape)

def _inputs_for_label(label: str) -> dict[str, Any]:
    return baseline_wide._inputs_for_label(label)

def _normalize_route_row(row: dict[str, Any]) -> dict[str, Any]:
    out = baseline_wide._normalize_route_row(row)
    if out.get('route_kind') not in {'specialized', 'general', 'fallback', 'coverage-only', 'none'}:
        out['route_kind'] = 'specialized' if out.get('selected_seed') else 'general'
    return out

def _baseline_route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    row = dict(baseline_wide.route_trace_for_contract_shapes((label,), force_fallback=force_fallback)[0])
    row['baseline_dispatcher_route'] = baseline_wide.route_for_contract_inputs(inputs)
    row['base_4247_route'] = base_4247.route_for_contract_inputs(inputs)
    return _normalize_route_row(row)

def _shape_spec_for_seed(seed_id: str, label: str) -> dict[str, Any]:
    if seed_id == SEED_D96_ID:
        return seed_d96.SHAPE_SPECS[label]
    if seed_id == SEED_D320_ID:
        return seed_d320.SHAPE_SPECS[label]
    if seed_id == SEED_D768_ID:
        return seed_d768.SHAPE_SPECS[label]
    if seed_id == SEED_K32_ID:
        return _K32_SHAPE_SPECS[label]
    raise KeyError(seed_id)

def _seed_entrypoint(seed_id: str) -> str:
    return {SEED_D96_ID: ROUTE_D96_ENTRYPOINT, SEED_D320_ID: ROUTE_D320_ENTRYPOINT, SEED_D768_ID: ROUTE_D768_ENTRYPOINT, SEED_K32_ID: ROUTE_K32_ENTRYPOINT}[seed_id]

def _guard_id(seed_id: str) -> str:
    return {SEED_D96_ID: 'c2eb_d96exact_exact_shape_guard', SEED_D320_ID: '8227_d320tail_exact_shape_guard', SEED_D768_ID: 'f533_d768_fused_exact_shape_guard', SEED_K32_ID: '8fcb_k32_split148_exact_shape_guard'}[seed_id]

def _producer_for_seed(seed_id: str, label: str) -> str:
    if seed_id == SEED_D96_ID:
        return seed_d96._producer_for_label(label)
    if seed_id == SEED_D320_ID:
        return seed_d320._producer_for_label(label)
    if seed_id == SEED_D768_ID:
        return ''.join(['m64_d768_fusedmerge_g', format(seed_d768.D768_GROUP_COUNT, '')])
    if seed_id == SEED_K32_ID:
        if label == seed_k32.Q8_K32_SHAPE:
            return 'q8_halfrow'
        if label == seed_k32.Q16_K32_SHAPE:
            return 'q16_rowld1'
        if label == seed_k32.Q32_K32_SHAPE:
            return 'q32_rowld_rows4'
        return 'inherited_q16_irregular_rowld1'
    raise KeyError(seed_id)

def _split_count_for_seed(seed_id: str, label: str) -> int | None:
    if seed_id == SEED_D96_ID:
        return seed_d96._split_count_for_label(label)
    if seed_id == SEED_D320_ID:
        return seed_d320._split_count_for_label(label)
    if seed_id == SEED_D768_ID:
        return seed_d768.D768_SPLIT_COUNT
    if seed_id == SEED_K32_ID:
        return seed_k32.K32_SPLIT_COUNT
    return None

def _specialized_trace_record(inputs: dict[str, Any], seed_id: str, label: str, *, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    spec = _shape_spec_for_seed(seed_id, label)
    targeted = dict(TARGETED_SEED_ROWS[label])
    return {'shape_key': label, 'selected_route': route_for_contract_inputs(inputs, **_candidate_enabled_kwargs(candidate_key)), 'selected_entrypoint': _seed_entrypoint(seed_id), 'selected_seed': seed_id, 'expected_seed': seed_id, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': _guard_id(seed_id), 'guard_condition': ''.join(['exact BF16 B=', format(spec['B'], ''), ' Q=', format(spec['Q'], ''), ' M=', format(spec['M'], ''), ' D=', format(spec['D'], ''), ' K=', format(spec['K'], ''), ' build=', format(spec.get('build', False), '')]), 'coverage': 'synthesized full82 seed route selected before the 8199-widecombine baseline', 'consumed_seed': seed_id, 'replaced_route': baseline_wide.route_for_contract_inputs(inputs), 'baseline_dispatcher_route': baseline_wide.route_for_contract_inputs(inputs), 'base_4247_route': base_4247.route_for_contract_inputs(inputs), 'producer': _producer_for_seed(seed_id, label), 'split_count': _split_count_for_seed(seed_id, label), 'targeted_seed_timing_backend': 'cupti', 'targeted_seed_kernel_ms': targeted['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': targeted['ratio_vs_flashlib'], 'row_selection': targeted, 'classification': 'seed-consumed', 'dispatcher_kernel_ms': None, 'shape_specific_kernel_ms': targeted['kernel_ms'], 'relative_speedup_vs_baseline': None}

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    selected_seed, label = _selected_seed_for_inputs(inputs, **_candidate_enabled_kwargs(candidate_key))
    if force_fallback and selected_seed is not None and (label is not None):
        row = _baseline_route_trace_record(inputs)
        row['expected_seed'] = selected_seed
        row['guard_id'] = 'forced_fallback_synthesized_portfolio_disabled'
        row['guard_condition'] = ''.join(['forced fallback to ', format(BASELINE_ID, ''), '; synthesized seed overlays disabled'])
        row['forced_disabled_seeds'] = _candidate_selected_seeds(candidate_key)
        row['classification'] = 'guard-miss'
        return _normalize_route_row(row)
    if not force_fallback and selected_seed is not None and (label is not None):
        return _normalize_route_row(_specialized_trace_record(inputs, selected_seed, label, candidate_key=candidate_key))
    return _baseline_route_trace_record(inputs)

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_from_shape(shape), force_fallback=force_fallback, candidate_key=candidate_key) for shape in selected]

def _timing_backends_for_reports(*reports: dict[str, Any]) -> list[str]:
    return baseline_wide._timing_backends_for_reports(*reports)

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return baseline_wide._rows_for_labels(report, labels)

def _seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> list[dict[str, Any]]:
    matrix = []
    enabled = _candidate_enabled_kwargs(candidate_key)
    for label in _candidate_target_shapes(candidate_key):
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        inputs = _inputs_for_label(label)
        selected_seed, _selected_label = _selected_seed_for_inputs(inputs, **enabled)
        targeted = TARGETED_SEED_ROWS[label]
        matrix.append({'shape_key': label, 'baseline_route': baseline_wide.route_for_contract_inputs(inputs), 'base_4247_route': base_4247.route_for_contract_inputs(inputs), 'candidate_route': route_for_contract_inputs(inputs, **enabled), 'selected_seed': selected_seed, 'candidate_id': _candidate_id(candidate_key), 'candidate_ms': candidate_ms, 'baseline_ms': baseline_ms, 'flashlib_ms': candidate_row.get('flashlib_ms'), 'metric_delta_ms': candidate_ms - baseline_ms if candidate_ms and baseline_ms else None, 'speedup_vs_baseline_dispatcher': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': candidate_row.get('ratio_vs_flashlib'), 'targeted_seed_kernel_ms': targeted['kernel_ms'], 'targeted_seed_ratio_vs_flashlib': targeted['ratio_vs_flashlib'], 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')})
    return matrix

def _frontmatter_seed_delta_matrix(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> list[dict[str, Any]]:
    rows = []
    for item in _seed_delta_matrix(candidate_report, baseline_report, candidate_key=candidate_key):
        rows.append({'shape_key': item['shape_key'], 'baseline_route': item['baseline_route'], 'candidate_deltas': [{'candidate_id': item['candidate_id'], 'selected_seed': item['selected_seed'], 'metric_delta': item['metric_delta_ms'], 'ratio_vs_flashlib': item['ratio_vs_flashlib'], 'timing_backend': item['timing_backend'] or 'cupti'}]})
    return rows

def _per_consumed_row_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    return {item['shape_key']: {'candidate_ms': item['candidate_ms'], 'baseline_dispatcher_ms': item['baseline_ms'], 'flashlib_ms': item['flashlib_ms'], 'speedup_vs_baseline_dispatcher': item['speedup_vs_baseline_dispatcher'], 'ratio_vs_flashlib': item['ratio_vs_flashlib'], 'candidate_route': item['candidate_route'], 'baseline_dispatcher_route': item['baseline_route'], 'base_4247_route': item['base_4247_route'], 'selected_seed': item['selected_seed'], 'targeted_seed_kernel_ms': item['targeted_seed_kernel_ms'], 'targeted_seed_ratio_vs_flashlib': item['targeted_seed_ratio_vs_flashlib'], 'candidate_passed': candidate_report.get('per_shape', {}).get(item['shape_key'], {}).get('passed'), 'baseline_passed': baseline_report.get('per_shape', {}).get(item['shape_key'], {}).get('passed')} for item in _seed_delta_matrix(candidate_report, baseline_report, candidate_key=candidate_key)}

def _annotate_route_trace(route_trace: list[dict[str, Any]], candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> list[dict[str, Any]]:
    annotated = []
    target_shape_set = set(_candidate_target_shapes(candidate_key))
    for row in route_trace:
        label = str(row.get('shape_key'))
        out = dict(row)
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        out['dispatcher_kernel_ms'] = candidate_ms
        out['baseline_dispatcher_kernel_ms'] = baseline_ms
        out['flashlib_ms'] = candidate_row.get('flashlib_ms')
        out['relative_speedup_vs_baseline'] = baseline_ms / candidate_ms if candidate_ms and baseline_ms else None
        out['route_changed_vs_baseline_dispatcher'] = out.get('selected_route') != out.get('baseline_dispatcher_route')
        ratio = candidate_row.get('ratio_vs_flashlib')
        if label in target_shape_set:
            if not out.get('selected_seed'):
                out['classification'] = 'guard-miss'
            elif isinstance(ratio, (float, int)) and ratio < 1.0:
                out['classification'] = 'kernel-slow'
            elif out['relative_speedup_vs_baseline'] is not None and out['relative_speedup_vs_baseline'] < 1.0:
                out['classification'] = 'kernel-slow'
            else:
                out['classification'] = 'seed-consumed'
        elif isinstance(ratio, (float, int)) and ratio < 1.0:
            out['classification'] = 'kernel-slow' if out.get('route_kind') == 'specialized' else 'fallback-slow'
        else:
            out['classification'] = out.get('classification', 'route-ok')
        annotated.append(_normalize_route_row(out))
    return annotated

def _below_flashlib_rows(report: dict[str, Any], *, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> list[dict[str, Any]]:
    trace_by_label = {str(row['shape_key']): row for row in route_trace_for_contract_shapes(candidate_key=candidate_key)}
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < 1.0:
            trace_row = trace_by_label.get(label, {})
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_route': trace_row.get('selected_route'), 'route_kind': trace_row.get('route_kind', 'unknown'), 'classification': 'kernel-slow' if trace_row.get('route_kind') == 'specialized' else 'fallback-slow'})
    return rows

def _benchmark_payload(candidate_report: dict[str, Any], baseline_report: dict[str, Any], *, use_cupti: bool, shape_labels, candidate_key: str=DEFAULT_CANDIDATE_KEY) -> dict[str, Any]:
    cfg = _candidate_config(candidate_key)
    consumed_labels = _candidate_target_shapes(candidate_key)
    selected_route_labels = tuple(dict.fromkeys((*baseline_wide.SELECTED_TARGET_SHAPES, *consumed_labels)))
    route_trace = _annotate_route_trace(route_trace_for_contract_shapes(shape_labels, candidate_key=candidate_key), candidate_report, baseline_report, candidate_key=candidate_key)
    below_flashlib = _below_flashlib_rows(candidate_report, candidate_key=candidate_key)
    candidate_metric = candidate_report['summary']['primary_mean']
    baseline_metric = baseline_report['summary']['primary_mean']
    return {'candidate_key': candidate_key, 'candidate_id': cfg['candidate_id'], 'selected_seeds': _candidate_selected_seeds(candidate_key), 'source_tasks': {SEED_D96_ID: 'weave-evolve-knn-build-c2eb / design_doc/active/weave_evolve_knn_build_round_104_4be7_d96exact.md', SEED_D320_ID: 'weave-evolve-knn-build-87c7 / design_doc/active/weave_evolve_knn_build_round_104_8227_d320tail.md', SEED_D768_ID: 'weave-evolve-knn-build-f533 / design_doc/active/weave_evolve_knn_build_round_105_4be7_d768fused.md', SEED_K32_ID: 'weave-evolve-knn-build-626a / design_doc/active/weave_evolve_knn_build_round_106_8fcb_k32split148.md'}, 'tflops': candidate_metric, 'baseline_tflops': baseline_metric, 'metric_delta': candidate_metric - baseline_metric if candidate_metric and baseline_metric else None, 'all_correct': candidate_report['summary']['all_correct'], 'baseline_all_correct': baseline_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'baseline_performance_comparable': baseline_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'baseline_invalid_performance_reason': baseline_report['summary']['invalid_performance_reason'], 'measured_entrypoint': cfg['benchmark_entrypoint'], 'baseline_entrypoint': BASELINE_ENTRYPOINT, 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'selected_route_labels': selected_route_labels, 'consumed_seed_labels': consumed_labels, 'guard_miss_audit_labels': GUARD_MISS_AUDIT_SHAPES, 'selected_route_rows': _rows_for_labels(candidate_report, selected_route_labels), 'baseline_selected_route_rows': _rows_for_labels(baseline_report, selected_route_labels), 'consumed_seed_rows': _rows_for_labels(candidate_report, consumed_labels), 'baseline_consumed_seed_rows': _rows_for_labels(baseline_report, consumed_labels), 'guard_miss_audit_rows': _rows_for_labels(candidate_report, GUARD_MISS_AUDIT_SHAPES), 'baseline_guard_miss_audit_rows': _rows_for_labels(baseline_report, GUARD_MISS_AUDIT_SHAPES), 'per_consumed_row_delta': _per_consumed_row_delta(candidate_report, baseline_report, candidate_key=candidate_key), 'seed_delta_matrix': _seed_delta_matrix(candidate_report, baseline_report, candidate_key=candidate_key), 'frontmatter_seed_delta_matrix': _frontmatter_seed_delta_matrix(candidate_report, baseline_report, candidate_key=candidate_key), 'targeted_seed_rows': TARGETED_SEED_ROWS, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'selected_candidate_dispatcher': cfg['candidate_id'], 'guard_plan': cfg['guard_plan'], 'route_modules': PRODUCTION_ROUTE_MODULES, 'route_trace': route_trace, 'forced_fallback_route_trace': route_trace_for_contract_shapes(shape_labels, force_fallback=True, candidate_key=candidate_key), 'route_trace_included': True, 'contract_summary': candidate_report['summary'], 'baseline_contract_summary': baseline_report['summary'], 'contract_performance': candidate_report['performance'], 'baseline_contract_performance': baseline_report['performance'], 'timing_backends': _timing_backends_for_reports(candidate_report, baseline_report), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'performance_coverage': 'partial' if below_flashlib else 'pass', 'coverage_only_routes': [], 'hot_bucket_blockers': below_flashlib, 'report': candidate_report, 'baseline_report': baseline_report}

def benchmark_candidate_portfolio(*, candidate_key: str=DEFAULT_CANDIDATE_KEY, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None) -> dict[str, Any]:
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=_candidate_kernel_fn(candidate_key))
    if baseline_report is None:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_wide)
    return _benchmark_payload(candidate_report, baseline_report, use_cupti=use_cupti, shape_labels=shape_labels, candidate_key=candidate_key)

def benchmark_candidate_d96_d320_only(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(candidate_key='d96_d320_only', **kwargs)

def benchmark_candidate_f533_non128_only(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(candidate_key='f533_non128_only', **kwargs)

def benchmark_candidate_k32_only(**kwargs) -> dict[str, Any]:
    return benchmark_candidate_portfolio(candidate_key='k32_only', **kwargs)

def benchmark_knn_build_dispatch_4247_non128_8199_c2eb_f533_8fcb_8227_full82_matrix_v1(*, use_cupti: bool=True, shape_labels=None, baseline_report: dict[str, Any] | None=None) -> dict[str, Any]:
    return benchmark_candidate_portfolio(candidate_key=DEFAULT_CANDIDATE_KEY, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report)

def benchmark_subset_matrix(*, use_cupti: bool=True, shape_labels=None, candidate_keys: tuple[str, ...]=CANDIDATE_KEYS) -> dict[str, Any]:
    baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=shape_labels, kernel_fn=candidate_baseline_wide)
    payloads = {key: benchmark_candidate_portfolio(candidate_key=key, use_cupti=use_cupti, shape_labels=shape_labels, baseline_report=baseline_report) for key in candidate_keys}
    baseline_metric = baseline_report['summary']['primary_mean']
    return {'matrix_id': 'f533_split148_matrix_over_8199_full82_v1', 'baseline_candidate_id': BASELINE_ID, 'baseline_entrypoint': BASELINE_ENTRYPOINT, 'baseline_tflops': baseline_metric, 'baseline_all_correct': baseline_report['summary']['all_correct'], 'baseline_report': baseline_report, 'candidate_keys': candidate_keys, 'candidate_dispatchers': CANDIDATE_DISPATCHERS, 'candidate_summaries': {key: {'candidate_id': payload['candidate_id'], 'measured_entrypoint': payload['measured_entrypoint'], 'selected_seeds': payload['selected_seeds'], 'tflops': payload['tflops'], 'metric_delta': payload['metric_delta'], 'all_correct': payload['all_correct'], 'performance_comparable': payload['performance_comparable'], 'performance_coverage': payload['performance_coverage'], 'hot_bucket_blocker_count': len(payload['hot_bucket_blockers'])} for key, payload in payloads.items()}, 'payloads': payloads, 'measured_shape_labels': 'all_canonical' if shape_labels is None else tuple(shape_labels), 'timing_backend_requested': 'cupti' if use_cupti else 'cuda_event', 'timing_backends': _timing_backends_for_reports(baseline_report, *(payload['report'] for payload in payloads.values())), 'route_trace_included': True, 'rank_objective': {key: {'metric': 'tflops', 'direction': 'maximize', 'value': payload['tflops'], 'baseline_value': baseline_metric, 'delta': payload['metric_delta'], 'denominator': 'full82_v9' if shape_labels is None else ''.join(['shape', format(len(tuple(shape_labels)), '')])} for key, payload in payloads.items()}}

def _denominator_label(shape_labels) -> str:
    if shape_labels is None:
        return ''.join(['full', format(len(eval_mod.CANONICAL_SHAPES), '')])
    return ''.join(['shape', format(len(tuple(shape_labels)), '')])

def write_benchmark_artifacts(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=None, candidate_key: str | None=None) -> dict[str, str]:
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    denom = _denominator_label(shape_labels)
    candidate_keys = CANDIDATE_KEYS if candidate_key is None else (candidate_key,)
    matrix = benchmark_subset_matrix(use_cupti=use_cupti, shape_labels=shape_labels, candidate_keys=candidate_keys)
    baseline_path = out_dir / ''.join([format(denom, ''), '_same_session_baseline_8199_widecombine_for_split148_matrix_v1.json'])
    summary_path = out_dir / ''.join([format(denom, ''), '_f533_split148_matrix_summary_v1.json'])
    paths: dict[str, str] = {'same_session_baseline_payload': str(baseline_path), 'matrix_summary': str(summary_path)}
    baseline_path.write_text(json.dumps({'candidate_id': BASELINE_ID, 'measured_entrypoint': BASELINE_ENTRYPOINT, 'measured_shape_labels': matrix['measured_shape_labels'], 'timing_backend_requested': matrix['timing_backend_requested'], 'timing_backends': matrix['timing_backends'], 'tflops': matrix['baseline_tflops'], 'all_correct': matrix['baseline_all_correct'], 'performance_comparable': matrix['baseline_report']['summary']['performance_comparable'], 'contract_summary': matrix['baseline_report']['summary'], 'contract_performance': matrix['baseline_report']['performance'], 'route_trace': baseline_wide.route_trace_for_contract_shapes(shape_labels), 'route_trace_included': True, 'report': matrix['baseline_report']}, indent=2, sort_keys=True) + '\n')
    for key, payload in matrix['payloads'].items():
        candidate_id = str(payload['candidate_id']).removeprefix('candidate_')
        candidate_path = out_dir / ''.join([format(denom, ''), '_dispatch_', format(candidate_id, ''), '.json'])
        route_trace_path = out_dir / ''.join([format(denom, ''), '_route_trace_', format(candidate_id, ''), '.json'])
        forced_trace_path = out_dir / ''.join([format(denom, ''), '_forced_fallback_trace_', format(candidate_id, ''), '.json'])
        seed_matrix_path = out_dir / ''.join([format(denom, ''), '_seed_delta_matrix_', format(candidate_id, ''), '.json'])
        candidate_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
        route_trace_path.write_text(json.dumps(payload['route_trace'], indent=2, sort_keys=True) + '\n')
        forced_trace_path.write_text(json.dumps(payload['forced_fallback_route_trace'], indent=2, sort_keys=True) + '\n')
        seed_matrix_path.write_text(json.dumps(payload['seed_delta_matrix'], indent=2, sort_keys=True) + '\n')
        paths[''.join([format(key, ''), '_candidate_payload'])] = str(candidate_path)
        paths[''.join([format(key, ''), '_route_trace'])] = str(route_trace_path)
        paths[''.join([format(key, ''), '_forced_fallback_trace'])] = str(forced_trace_path)
        paths[''.join([format(key, ''), '_seed_delta_matrix'])] = str(seed_matrix_path)
    summary_payload = {key: value for key, value in matrix.items() if key not in {'payloads', 'baseline_report'}}
    summary_payload['candidate_payload_paths'] = {key: paths[''.join([format(key, ''), '_candidate_payload'])] for key in matrix['payloads']}
    summary_payload['same_session_baseline_payload'] = str(baseline_path)
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + '\n')
    return paths
