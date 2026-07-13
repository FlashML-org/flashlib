"""Build-broad bucket dispatcher for the dbd7 auto-tuning continuation.

Minimum target architecture: sm_100a. This additive candidate keeps the
17b8/1074 full82 dispatcher as the fallback and routes exact BF16 build
low-floor rows through existing Weave seed families:

* v20 split/tcgen05 build path for K10/K12/K20 D128 build rows.
* v25 over-32 split/tcgen05 path for K48 D128 build rows.

No external runtime fallback is used; FlashLib/PyTorch remain contract-harness
references only.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable
from .. import _dispatch_runtime as eval_mod
from . import knn_build_dispatch_17b8_lowmargin_1074_full82_v1 as base17b8
from . import knn_build_over32_topk_knn_build_dispatch_slurm_0610_6329_v25 as over32_v25
from . import knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20 as build_v20
MODULE = 'loom.examples.weave.knn_build_dispatch_dbd7_build_broad_8a78_v1'
K10_TARGET_SHAPES = ('build_qm2048_d128_k10', 'build_tail_b1_q1536_m1536_d128_k10')
V20_TARGET_SHAPES = ('build_k_sweep_qm1024_k12', 'build_k_sweep_qm1024_k20', 'build_k_sweep_qm4096_k12')
OVER32_TARGET_SHAPES = ('build_over32_stress_qm2048_k48', 'build_over32_stress_qm4096_k48')
TARGET_SHAPES = (*K10_TARGET_SHAPES, *V20_TARGET_SHAPES, *OVER32_TARGET_SHAPES)
TARGET_SHAPE_SET = set(TARGET_SHAPES)
K10_TARGET_SHAPE_SET = set(K10_TARGET_SHAPES)
V20_TARGET_SHAPE_SET = set(V20_TARGET_SHAPES)
OVER32_TARGET_SHAPE_SET = set(OVER32_TARGET_SHAPES)
SEED_K10_ID = 'dbd7_8a78_fixedbuild_k10_v2'
SEED_V20_ID = 'dbd7_8a78_v20_build_broad'
SEED_OVER32_ID = 'dbd7_8a78_over32_k48_v25'
BASE_17B8_ID = base17b8.CANDIDATE_LOWMARGIN_1074
CANDIDATE_ID = 'candidate_dbd7_build_broad_8a78_v1'
ROUTE_K10_BUILD = 'loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2:launch_from_contract_inputs'
ROUTE_V20_BUILD = 'loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20:launch_from_contract_inputs'
ROUTE_OVER32_K48 = 'loom.examples.weave.knn_build_over32_topk_knn_build_dispatch_slurm_0610_6329_v25:launch_from_contract_inputs'
ROUTE_BASE_17B8 = ''.join([format(base17b8.MODULE, ''), ':launch_from_contract_inputs'])
PRODUCTION_ROUTE_MODULES = _decode_capture(_json_loads('{"__dict_items__": [["large_square_k20k32", "loom.examples.weave.knn_build_large_square_k20k32_a989_v1"], ["over64_k96", "loom.examples.weave.knn_build_over64_k96_a989_v1"], ["baseline_7c3a_rag_k10", "loom.examples.weave.knn_build_rag_frontier_4b5c_v1:k10"], ["rag_frontier_7399_k10", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k10_s72"], ["rag_frontier_7399_k32_replaced", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k32_s72_g8_fusedmerge"], ["rag_frontier_4fbf_k32", "loom.examples.weave.knn_build_rag_frontier_4fbf_v7:k32_s72_g24_tailinf_fused"], ["rect_smallq_largem_d15e", "loom.examples.weave.knn_build_rect_smallq_largem_ff59_d15e_v1:split16"], ["baseline_7c3a_policy", "loom.examples.weave.knn_build_dispatch_b6d4_d15e_fd02_v1:baseline_7c3a_policy"], ["fallback", "loom.examples.weave.knn_build_dispatch_split72_4e09_de1a_3dc7_v48"], ["dim_d64_73a9", "loom.examples.weave.knn_build_dim_midk_73a9_v1:d64_split_s8"], ["current_exact_k32_dispatcher", "loom.examples.weave.knn_build_dispatch_4fbf_7399_d15e_full55_bad5_v1:launch_from_contract_inputs"], ["base_7399_d15e_dispatcher", "loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:launch_from_contract_inputs"], ["rag_frontier_7399_k32", "loom.examples.weave.knn_build_rag_frontier_7399_v1:k32_s72_g8_fusedmerge"], ["dim_d256_df2f", "loom.examples.weave.knn_build_dim_midk_df2f_v1:d256_split_s8"], ["dim_fp16_d128_df2f", "loom.examples.weave.knn_build_dim_midk_df2f_v1:fp16_d128_split_s8"], ["base_dispatch", "loom.examples.weave.knn_build_dispatch_7399_d15e_full55_v1:launch_from_contract_inputs"], ["rect_intermediate_4452_s8", "loom.examples.weave.knn_build_rect_intermediate_frontier_6a73_4452_v2:rect_s8_k10_cached"], ["base_champion_6b59", "loom.examples.weave.knn_build_dispatch_7399_d15e_df2f_full55_v1:launch_from_contract_inputs"], ["base_k32_d64_62b1", "loom.examples.weave.knn_build_dispatch_4fbf_7399_d15e_73a9_full55_v1:launch_from_contract_inputs"], ["default_k96_a330", "loom.examples.weave.knn_build_over64_k96_a989_v1"], ["large_tail_a4f6", "loom.examples.weave.knn_build_large_tail_frontier_6a73_v1:split4_k20"], ["midk_81aa_q2048_k24_k28", "loom.examples.weave.knn_build_dim_midk_bad5_midkcleanup_v1:midk_k24_k28_s8"], ["midk_9b2c_q4096_k28", "loom.examples.weave.knn_build_dim_midk_bad5_k24k28_v1:k28_q4096_s4_unordered_exact"], ["base_f552", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f552_v1:launch_from_contract_inputs"], ["midk_bad5_k64split8", "loom.examples.weave.knn_build_dim_midk_bad5_k64split8_v1:k64_q2048_s8_tailinf"], ["base_e51c", "loom.examples.weave.knn_build_dispatch_selected_portfolio_e51c_v1:launch_from_contract_inputs"], ["midk_f8c3_q4096_k64_split8_a194", "loom.examples.weave.knn_build_dim_midk_f8c3_q4096k64split_v1:q4096_k64_tailinf_split8"], ["base_f8c3", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f8c3_v1:launch_from_contract_inputs"], ["lowk_b193_q512_s4", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"], ["lowk_b193_q1024_k16_s16", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q1024_k16_s16"], ["large_square_5407_q8192_k32_s2", "loom.examples.weave.knn_build_large_square_k32_8a83_v1:q8192_k32_split2"], ["base_f853", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f853_v1:launch_from_contract_inputs"], ["lowk_b193_q512_k4_k5_k6_s4", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"], ["base_f16b", "loom.examples.weave.knn_build_dispatch_selected_portfolio_f16b_v1:launch_from_contract_inputs"], ["rag_microbatch_b2ec_s72_g8", "loom.examples.weave.knn_build_rag_microbatch_4a72_v1:launch_from_contract_inputs"], ["base_4a72", "loom.examples.weave.knn_build_dispatch_selected_portfolio_4a72_v1:launch_from_contract_inputs"], ["rag_m64_s128_0c69", "loom.examples.weave.knn_build_rag_microbatch_m64_d4f7_v1:launch_from_contract_inputs"], ["rag_s144_g12_cta1_059f", "loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs"], ["rag_s144_g8_cta1_4982_read_ref_parameterized", "loom.examples.weave.knn_build_rag_microbatch_4a72_v2:launch_from_contract_inputs"], ["base_397b", "loom.examples.weave.knn_build_dispatch_selected_portfolio_397b_v1:launch_from_contract_inputs"], ["d64_fdd7_aa88_v2", "loom.examples.weave.knn_build_d64_build_aa88_v2:launch_from_contract_inputs"], ["base_8700", "loom.examples.weave.knn_build_dispatch_rag_seed_portfolio_8700_v1:launch_from_contract_inputs(portfolio_id=all_m64_s128)"], ["rect_d64_cf49_v3_9138", "loom.examples.weave.knn_build_rect_d64_cf49_v3:launch_from_contract_inputs"], ["q1_mbucket_aa88_q1m_v3_bcb3", "loom.examples.weave.knn_build_ragonline_mbucket_aa88_q1m_v3:launch_from_contract_inputs"], ["over64_k96_a2f8_v1_generated_v8", "loom.examples.weave.knn_build_over64_k96_a2f8_v1:_launch_over64_k96_split_path"], ["base_e3de", "loom.examples.weave.knn_build_dispatch_d64_fdd7_e3de_v1:launch_from_contract_inputs"], ["non128_frontier_8199_widecombine_v1", "loom.examples.weave.knn_build_non128_frontier_8199_widecombine_v1:launch_from_contract_inputs"], ["base_4247", "loom.examples.weave.knn_build_dispatch_e3de_9138_bcb3_4247_v1:launch_from_contract_inputs"], ["rag_microbucket_k32_8fcb_split148_v1_b3e0_sm148", "loom.examples.weave.knn_build_rag_microbucket_k32_8fcb_split148_v1:launch_from_contract_inputs"], ["rag_microbucket_k32_2e8e_q16split148_v1_b3e0_q16_s148", "loom.examples.weave.knn_build_rag_microbucket_k32_2e8e_q16split148_v1:launch_from_contract_inputs"], ["non128_frontier_3d5a_cachedmerge_v1", "loom.examples.weave.knn_build_non128_frontier_3d5a_cachedmerge_v1:launch_from_contract_inputs"], ["over64_k96_exactall_229a_v1_b6c4", "loom.examples.weave.knn_build_over64_k96_exactall_229a_v1:launch_from_contract_inputs"], ["knn_build_midk_k11k13_e080_v1", "loom.examples.weave.knn_build_midk_k11k13_e080_v1:launch_from_contract_inputs"], ["ragonline_mbucket_4fc7_q1m262_v1_980c", "loom.examples.weave.knn_build_ragonline_mbucket_4fc7_q1m262_v1:launch_from_contract_inputs"], ["baseline_8199_widecombine_full82_v1", "loom.examples.weave.knn_build_dispatch_4247_non128_8199_widecombine_full82_v1:launch_from_contract_inputs"], ["k30_q4096_6998_warpselect_v1", "loom.examples.weave.knn_build_k30_q4096_6998_warpselect_v1:launch_from_contract_inputs"], ["rag_stream_k10_direct_split72_6998_v1", "loom.examples.weave.knn_build_rag_online_stream_split72_4e09_v1:launch_from_contract_inputs"], ["rect_d64_23be_unordered_v1", "loom.examples.weave.knn_build_rect_d64_23be_unordered_v1:launch_from_contract_inputs"], ["residual_19b3_ed1c_portfolio_6998", "loom.examples.weave.knn_build_dispatch_c142_3505_q32rowld_19b3_v1:launch_from_contract_inputs"], ["candidate_q16split148_cachedmerge_k96exactall_e080_q1m262_over_8199_full82_v1", "loom.examples.weave.knn_build_dispatch_4247_non128_8199_3d5a_2e8e_full82_synth_v1:launch_from_contract_inputs"], ["rect_d128_k20_q1536_9b9f_d555_b8c7_v1", "loom.examples.weave.knn_build_rect_d128_k20_d555_b8c7_v1:launch_from_contract_inputs"], ["rag_microbatch_k10_q4q64_m64_3505_d555_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4q64_d555_v1:launch_from_contract_inputs"], ["rag_microbucket_faeb_q4q64_k10_69d6_v1", "loom.examples.weave.knn_build_rag_microbucket_faeb_v1:launch_from_contract_inputs"], ["candidate_066c_ragmicro_q4q64_3505_full82_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4q64_d555_v1:launch_from_contract_inputs"], ["candidate_d555_direct_residual_seeds_full82_v1", "loom.examples.weave.knn_build_dispatch_d555_residual_seed_synth_full82_v1:launch_from_contract_inputs"], ["rag_microbatch_k10_q4_m64_s144_g12_17b8_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4_m64s144_17b8_v1:launch_from_contract_inputs"], ["rag_microbatch_q4_k10_s144_17b8_v1", "loom.examples.weave.knn_build_rag_microbatch_q4_s144_17b8_v1:launch_from_contract_inputs"], ["rag_microbatch_k10_q4_s144_g12_d555_v1", "loom.examples.weave.knn_build_rag_microbatch_k10_q4_s144_d555_v1:launch_from_contract_inputs"], ["candidate_066c_69d6_plus_b8c7_full82_v1", "loom.examples.weave.knn_build_dispatch_066c_b8c7_69d6_q4_portfolio_full82_v1:launch_from_contract_inputs"], ["lowmargin_1074_k1k24k30_v1", "loom.examples.weave.knn_build_lowmargin_1074_k1k24k30_v1:launch_from_contract_inputs"], ["lowk_q512_k1_s4_1074", "loom.examples.weave.knn_build_lowk_f8c3_q512_q1024_v1:q512_lowk_s4"], ["k24_q4096_warpselect_1074", "loom.examples.weave.knn_build_lowmargin_1074_k1k24k30_v1:k24_q4096_warpselect"], ["dbd7_8a78_fixedbuild_k10_v2", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2:launch_from_contract_inputs"], ["dbd7_8a78_v20_build_broad", "loom.examples.weave.knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20:launch_from_contract_inputs"], ["dbd7_8a78_over32_k48_v25", "loom.examples.weave.knn_build_over32_topk_knn_build_dispatch_slurm_0610_6329_v25:launch_from_contract_inputs"], ["candidate_17b8_lowmargin_1074_full82_v1", "loom.examples.weave.knn_build_dispatch_17b8_lowmargin_1074_full82_v1:launch_from_contract_inputs"]]}'))
SOURCE_TASKS = _decode_capture(_json_loads('{"__dict_items__": [["dbd7_8a78_fixedbuild_k10_v2", "weave-evolve prior fixed-build K10 lineage / loom/examples/weave/knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2.py"], ["dbd7_8a78_v20_build_broad", "weave-evolve prior v20 lineage / loom/examples/weave/knn_build_evolve_7bfc_split_cg2_u2_smallmedfan_rag7_k10merge_stage1batch_cond4_k5merge4tree_vmin_maxtree_k5tree_mintree_k10s4s7cache_t32r32_k10mintree_fixedbuild_dispatch_v2_k32split_v20.py"], ["dbd7_8a78_over32_k48_v25", "weave-evolve prior v25 over32 lineage / loom/examples/weave/knn_build_over32_topk_knn_build_dispatch_slurm_0610_6329_v25.py"], ["candidate_17b8_lowmargin_1074_full82_v1", "generalize-auto-tuning-knn-build-a444 / loom/examples/weave/knn_build_dispatch_17b8_lowmargin_1074_full82_v1.py"]]}'))

class _TraceTensor:

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype if dtype.startswith('torch.') else ''.join(['torch.', format(dtype, '')])

def _verify_export_ir() -> Any:
    verify_kernel = os.environ.get('LOOM_KNN_DBD7_BUILD_BROAD_VERIFY_KERNEL')
    if verify_kernel == 'v20_stage1_k12':
        return build_v20.stage1_k12_ir
    if verify_kernel == 'v20_stage1_k20':
        return build_v20.stage1_k20_ir
    if verify_kernel == 'v20_stage1_k12_unordered':
        return build_v20.stage1_k12_unordered_ir
    if verify_kernel == 'v20_merge_k12_s16':
        return build_v20.merge_k12_s16_ir
    if verify_kernel == 'v20_merge_k12_unordered':
        return build_v20.merge_k12_unordered_ir
    if verify_kernel == 'v20_merge_k20_s16':
        return build_v20.merge_k20_s16_ir
    if verify_kernel == 'v20_merge_k20_s8':
        return build_v20.merge_k20_s8_ir
    if verify_kernel == 'k10_stage1':
        return build_v20.parent.stage1_ir
    if verify_kernel == 'k10_merge_s7_cache':
        return build_v20.parent.parent.parent_cached.merge_k10_s7_cache_ir
    if verify_kernel == 'over32_stage1_k48':
        return over32_v25.stage1_k48_over32_ir
    if verify_kernel == 'over32_merge_k48':
        return over32_v25.merge_k48_over32_ir
    return base17b8.ir
ir = _decode_capture(_json_loads('{"__ir__": "knn_build_rag_microbatch_m64_d4f7_stage1", "arg_keys": ["query", "database", "query_sq", "database_sq", "partial_dists", "partial_indices", "B", "Q", "M", "K", "num_q_tiles", "db_tiles_per_split", "split_count", "total_work"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 91392, "constants": [], "cta_group": 1, "threads": 512}'))

def _dtype_name(inputs: dict[str, Any], name: str='query') -> str:
    tensor = inputs.get(name)
    if tensor is not None and hasattr(tensor, 'dtype'):
        return str(tensor.dtype).replace('torch.', '')
    return str(inputs.get('dtype', '')).replace('torch.', '')

def _label_can_hit(inputs: dict[str, Any], target_labels: set[str]) -> bool:
    label = inputs.get('label')
    return label is None or str(label) in target_labels

def _is_bf16_d128_build(inputs: dict[str, Any]) -> bool:
    return bool(inputs.get('build', False)) and int(inputs.get('B', -1)) == 1 and (int(inputs.get('Q', -1)) == int(inputs.get('M', -2))) and (int(inputs.get('D', -1)) == 128) and (_dtype_name(inputs, 'query') == 'bfloat16') and (_dtype_name(inputs, 'database') in ('', 'bfloat16'))

def _eligible_k10_build(inputs: dict[str, Any]) -> bool:
    if not (_label_can_hit(inputs, K10_TARGET_SHAPE_SET) and _is_bf16_d128_build(inputs)):
        return False
    q = int(inputs.get('Q', -1))
    k = int(inputs.get('K', -1))
    return (q, k) in {(2048, 10), (1536, 10)}

def _eligible_v20_build(inputs: dict[str, Any]) -> bool:
    if not (_label_can_hit(inputs, V20_TARGET_SHAPE_SET) and _is_bf16_d128_build(inputs)):
        return False
    q = int(inputs.get('Q', -1))
    k = int(inputs.get('K', -1))
    return (q, k) in {(1024, 12), (1024, 20), (4096, 12)}

def _eligible_over32_k48(inputs: dict[str, Any]) -> bool:
    return _label_can_hit(inputs, OVER32_TARGET_SHAPE_SET) and over32_v25._eligible_over32_build(inputs)

def route_for_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> str:
    if not force_fallback and _eligible_k10_build(inputs):
        return ROUTE_K10_BUILD
    if not force_fallback and _eligible_v20_build(inputs):
        return ROUTE_V20_BUILD
    if not force_fallback and _eligible_over32_k48(inputs):
        return ROUTE_OVER32_K48
    return base17b8.route_for_contract_inputs(inputs, force_fallback=force_fallback)

def _launch_route(inputs: dict[str, Any], route: str) -> None:
    if route == ROUTE_V20_BUILD:
        build_v20.launch_from_contract_inputs(inputs)
        return
    if route == ROUTE_K10_BUILD:
        build_v20.parent.launch_from_contract_inputs(inputs)
        return
    if route == ROUTE_OVER32_K48:
        over32_v25._launch_over32_split_path(inputs)
        return
    base17b8.launch_from_contract_inputs(inputs)

def launch_from_contract_inputs(inputs: dict[str, Any], *, force_fallback: bool=False) -> None:
    _launch_route(inputs, route_for_contract_inputs(inputs, force_fallback=force_fallback))

def candidate_dbd7_build_broad_8a78_v1(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs)

def candidate(inputs: dict[str, Any]) -> None:
    candidate_dbd7_build_broad_8a78_v1(inputs)

def candidate_base_17b8(inputs: dict[str, Any]) -> None:
    base17b8.launch_from_contract_inputs(inputs)

def candidate_force_fallback(inputs: dict[str, Any]) -> None:
    launch_from_contract_inputs(inputs, force_fallback=True)

def evaluate_contract(*, shapes=None, correctness: bool=True, benchmark: bool=True, kernel_fn: Callable[[dict[str, Any]], Any]=candidate) -> dict[str, Any]:
    return eval_mod.evaluate(kernel_fn, shapes=shapes, correctness=correctness, benchmark=benchmark)

def _select_contract_shapes(shape_labels):
    return base17b8._select_contract_shapes(shape_labels)

def _benchmark_shapes(shape_labels, *, time_flashlib: bool) -> list[dict[str, Any]]:
    selected = _select_contract_shapes(TARGET_SHAPES if shape_labels is None else shape_labels)
    out = []
    for shape in selected:
        params = dict(shape['params'])
        params['time_flashlib'] = bool(time_flashlib)
        out.append({'label': shape['label'], 'params': params})
    return out

def _set_bench_backend(use_cupti: bool):
    previous = eval_mod.CONTRACT.bench.get('use_cupti', True)
    eval_mod.CONTRACT.bench['use_cupti'] = bool(use_cupti)
    return previous

def _run_with_timing_backend(*, use_cupti: bool, shape_labels=TARGET_SHAPES, kernel_fn: Callable[[dict[str, Any]], Any]=candidate, correctness: bool=True, benchmark: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    previous = _set_bench_backend(use_cupti)
    try:
        return evaluate_contract(shapes=_benchmark_shapes(shape_labels, time_flashlib=time_flashlib), correctness=correctness, benchmark=benchmark, kernel_fn=kernel_fn)
    finally:
        eval_mod.CONTRACT.bench['use_cupti'] = previous

def compile_and_launch_knn_build(*, shape_labels=TARGET_SHAPES, benchmark: bool=False) -> dict[str, Any]:
    report = evaluate_contract(shapes=_select_contract_shapes(shape_labels), correctness=True, benchmark=benchmark)
    passed = bool(report.get('summary', {}).get('all_correct', False))
    report['passed'] = passed
    report['all_pass'] = passed
    return report

def _trace_inputs_for_shape(shape: dict[str, Any]) -> dict[str, Any]:
    params = dict(shape['params'])
    dtype = str(params.get('dtype', 'bfloat16'))
    return {'label': shape['label'], 'B': int(params['B']), 'Q': int(params['Q']), 'M': int(params['M']), 'D': int(params['D']), 'K': int(params['K']), 'dtype': dtype, 'build': bool(params.get('build', False)), 'query': _TraceTensor(dtype), 'database': _TraceTensor(dtype)}

def _expected_seed(inputs: dict[str, Any]) -> str | None:
    if _eligible_k10_build(inputs):
        return SEED_K10_ID
    if _eligible_v20_build(inputs):
        return SEED_V20_ID
    if _eligible_over32_k48(inputs):
        return SEED_OVER32_ID
    return None

def _selected_entrypoint(route: str) -> str:
    if route == ROUTE_V20_BUILD:
        return ROUTE_V20_BUILD
    if route == ROUTE_K10_BUILD:
        return ROUTE_K10_BUILD
    if route == ROUTE_OVER32_K48:
        return ROUTE_OVER32_K48
    return ROUTE_BASE_17B8

def _route_trace_record(inputs: dict[str, Any], *, force_fallback: bool=False) -> dict[str, Any]:
    label = str(inputs.get('label'))
    expected_seed = _expected_seed(inputs)
    route = route_for_contract_inputs(inputs, force_fallback=force_fallback)
    baseline_route = base17b8.route_for_contract_inputs(inputs)
    if expected_seed is None or force_fallback:
        row = dict(base17b8._route_trace_record(inputs, force_fallback=force_fallback))
        row['expected_seed'] = expected_seed
        row['baseline_17b8_route'] = baseline_route
        row['candidate_guard_status'] = 'forced_fallback' if force_fallback else 'guard_miss'
        if force_fallback and expected_seed is not None:
            row['guard_id'] = ''.join(['forced_fallback_', format(expected_seed, ''), '_disabled'])
            row['guard_condition'] = ''.join(['forced fallback to 17b8; ', format(expected_seed, ''), ' disabled'])
            row['classification'] = 'guard-miss'
        return base17b8._normalize_route_row(row)
    if expected_seed == SEED_K10_ID:
        guard_id = 'dbd7_8a78_fixedbuild_k10_exact_guard'
    elif expected_seed == SEED_V20_ID:
        guard_id = 'dbd7_8a78_v20_build_exact_guard'
    else:
        guard_id = 'dbd7_8a78_over32_k48_exact_guard'
    return base17b8._normalize_route_row({'shape_key': label, 'selected_route': route, 'selected_entrypoint': _selected_entrypoint(route), 'selected_seed': expected_seed, 'expected_seed': expected_seed, 'route_kind': 'specialized', 'route_source': 'shape-specific-seed', 'guard_id': guard_id, 'guard_condition': 'exact BF16 build B=1 Q=M D=128 build-broad bucket guard', 'baseline_17b8_route': baseline_route, 'replaced_route': baseline_route, 'classification': 'seed-consumed'})

def route_trace_for_contract_shapes(shape_labels=None, *, force_fallback: bool=False) -> list[dict[str, Any]]:
    selected = eval_mod.CANONICAL_SHAPES if shape_labels is None else _select_contract_shapes(shape_labels)
    return [_route_trace_record(_trace_inputs_for_shape(shape), force_fallback=force_fallback) for shape in selected]

def _rows_for_labels(report: dict[str, Any], labels: tuple[str, ...]) -> dict[str, Any]:
    return {label: dict(report.get('per_shape', {}).get(label, {})) for label in labels}

def _per_shape_delta(candidate_report: dict[str, Any], baseline_report: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for label in TARGET_SHAPES:
        candidate_row = candidate_report.get('per_shape', {}).get(label, {})
        baseline_row = baseline_report.get('per_shape', {}).get(label, {})
        candidate_ms = candidate_row.get('kernel_ms')
        baseline_ms = baseline_row.get('kernel_ms')
        flashlib_ms = candidate_row.get('flashlib_ms') or baseline_row.get('flashlib_ms')
        rows[label] = {'candidate_ms': candidate_ms, 'baseline_17b8_ms': baseline_ms, 'flashlib_ms': flashlib_ms, 'candidate_tflops': candidate_row.get('tflops'), 'baseline_17b8_tflops': baseline_row.get('tflops'), 'speedup_vs_17b8': baseline_ms / candidate_ms if candidate_ms and baseline_ms else None, 'ratio_vs_flashlib': flashlib_ms / candidate_ms if candidate_ms and flashlib_ms else None, 'passed': candidate_row.get('passed'), 'timing_backend': candidate_row.get('timing_backend') or baseline_row.get('timing_backend')}
    return rows

def _below_flashlib_rows(report: dict[str, Any], *, floor: float) -> list[dict[str, Any]]:
    rows = []
    for label, row in sorted(report.get('per_shape', {}).items()):
        ratio = row.get('ratio_vs_flashlib')
        if isinstance(ratio, (float, int)) and ratio < floor:
            rows.append({'shape_key': label, 'kernel_ms': row.get('kernel_ms'), 'flashlib_ms': row.get('flashlib_ms'), 'ratio_vs_flashlib': ratio, 'selected_seed': _expected_seed(_trace_inputs_for_shape(_select_contract_shapes((label,))[0]))})
    return rows

def benchmark_candidate_dbd7_build_broad_8a78_v1(*, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, Any]:
    labels = tuple((str(label) for label in shape_labels))
    candidate_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate, correctness=True, benchmark=True, time_flashlib=time_flashlib)
    baseline_report = None
    if run_baseline:
        baseline_report = _run_with_timing_backend(use_cupti=use_cupti, shape_labels=labels, kernel_fn=candidate_base_17b8, correctness=True, benchmark=True, time_flashlib=time_flashlib)
    candidate_mean = candidate_report['summary']['primary_mean']
    payload: dict[str, Any] = {'candidate_id': CANDIDATE_ID, 'selected_seeds': (SEED_K10_ID, SEED_V20_ID, SEED_OVER32_ID), 'source_tasks': SOURCE_TASKS, 'all_correct': candidate_report['summary']['all_correct'], 'performance_comparable': candidate_report['summary']['performance_comparable'], 'invalid_performance_reason': candidate_report['summary']['invalid_performance_reason'], 'tflops': candidate_mean, 'measured_entrypoint': ''.join([format(MODULE, ''), ':benchmark_candidate_dbd7_build_broad_8a78_v1']), 'candidate_entrypoint': ''.join([format(MODULE, ''), ':launch_from_contract_inputs']), 'measured_shape_labels': labels, 'timing_backend': 'cupti' if use_cupti else 'cuda_event', 'use_cupti': use_cupti, 'time_flashlib': time_flashlib, 'denominator': 'dbd7_build_broad_low_floor_exact7', 'route_trace': route_trace_for_contract_shapes(labels), 'forced_fallback_route_trace': route_trace_for_contract_shapes(labels, force_fallback=True), 'route_modules': PRODUCTION_ROUTE_MODULES, 'contract_summary': candidate_report['summary'], 'contract_performance': candidate_report['performance'], 'contract_correctness': candidate_report['correctness'], 'selected_route_rows': _rows_for_labels(candidate_report, labels), 'hot_bucket_blockers': _below_flashlib_rows(candidate_report, floor=1.05), 'rank_objective': {'metric': 'tflops', 'direction': 'maximize', 'value': candidate_mean, 'valid_measurement_count': candidate_report['performance']['valid_measurement_count'], 'comparable': candidate_report['performance']['comparable']}, 'report': candidate_report}
    if baseline_report is not None:
        baseline_mean = baseline_report['summary']['primary_mean']
        payload.update({'baseline_candidate_id': BASE_17B8_ID, 'baseline_entrypoint': ''.join([format(base17b8.MODULE, ''), ':benchmark_candidate_17b8_lowmargin_1074_full82_v1']), 'baseline_tflops': baseline_mean, 'metric_delta_vs_17b8': candidate_mean - baseline_mean if candidate_mean is not None and baseline_mean is not None else None, 'baseline_contract_summary': baseline_report['summary'], 'baseline_contract_performance': baseline_report['performance'], 'baseline_selected_route_rows': _rows_for_labels(baseline_report, labels), 'per_shape_delta_vs_17b8': _per_shape_delta(candidate_report, baseline_report)})
    return payload

def write_benchmark_artifact(artifact_dir: str | Path, *, use_cupti: bool=True, shape_labels=TARGET_SHAPES, run_baseline: bool=True, time_flashlib: bool=True) -> dict[str, str]:
    payload = benchmark_candidate_dbd7_build_broad_8a78_v1(use_cupti=use_cupti, shape_labels=shape_labels, run_baseline=run_baseline, time_flashlib=time_flashlib)
    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'build_broad_8a78_v1.json'
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return {'candidate_payload': str(path)}
