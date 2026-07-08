"""D256/Q128/M131072/K64 fused hierarchical-ownership kNN seed.

Minimum target architecture: sm_100a.  The inherited round-36 tcgen05/TMEM
producer emits 512 exact sorted top-64 lists.  Eight consumer warps own one
64-list group each inside a CTA per query row, hand their exact group top-64
lists through CTA shared memory, and let warp 0 write the final exact top-64
directly to caller-owned outputs.  The fused consumer removes the incumbent's
single 512-list serial merge without changing its producer or split-M point.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
import math
from typing import Any
from .. import _dispatch_runtime as contract
from . import knn_search_extendedk_lowd_d256_dispatch0610_r3_extk_v1 as incumbent
from . import knn_search_lowd_d256_mma64_over48e9_0612_r34_48e9_v1 as producer
THREADS = producer.THREADS
BLOCK_Q = producer.BLOCK_Q
BLOCK_M = producer.BLOCK_M
D_STATIC = producer.D_STATIC
K_MAX = producer.K_MAX
SPLIT_M = producer.D256_MMA_SPLIT_M
POST_MMA_COL_COHORTS = producer.MMA_POST_MMA_COL_COHORTS
PARTIAL_LISTS = SPLIT_M * POST_MMA_COL_COHORTS
MMA_SMEM_BYTES = producer.MMA_SMEM_BYTES
HIERARCHICAL_GROUPS = 8
LISTS_PER_GROUP = PARTIAL_LISTS // HIERARCHICAL_GROUPS
WARP_LANES = 32
HEADS_PER_GROUP_LANE = LISTS_PER_GROUP // WARP_LANES
WARPS_PER_CTA = HIERARCHICAL_GROUPS
GROUP_SMEM_ENTRIES = HIERARCHICAL_GROUPS * K_MAX
FUSED_SMEM_BYTES = GROUP_SMEM_ENTRIES * 8
TARGET_SHAPE = 'glm5_rag_q128_m131072_d256_k64'
TARGET_LABELS = (TARGET_SHAPE,)
_SHAPES_BY_LABEL = _decode_capture(_json_loads('{"__dict_items__": [["ksweep_q4096_m20000_d128_k64", {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610313], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["glm5_rag_q128_m131072_d256_k64", {"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 610406], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d33_q128_m131072_k10", {"__dict_items__": [["label", "blind_0622_dyn_d33_q128_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 33], ["K", 10], ["dtype", "bfloat16"], ["seed", 611005], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d80_q256_m65536_k10", {"__dict_items__": [["label", "blind_0622_dyn_d80_q256_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 65536], ["D", 80], ["K", 10], ["dtype", "bfloat16"], ["seed", 611006], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d160_q64_m131072_k10", {"__dict_items__": [["label", "blind_0622_dyn_d160_q64_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 160], ["K", 10], ["dtype", "bfloat16"], ["seed", 611007], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d640_q32_m32768_k10", {"__dict_items__": [["label", "blind_0622_dyn_d640_q32_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 640], ["K", 10], ["dtype", "bfloat16"], ["seed", 611008], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_b2_q96_m196608_d128_k10", {"__dict_items__": [["label", "blind_0622_b2_q96_m196608_d128_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 96], ["M", 196608], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611009], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_tail_q1025_m65537_d128_k10", {"__dict_items__": [["label", "blind_0622_tail_q1025_m65537_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1025], ["M", 65537], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611012], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d64_q128_m131072_k64", {"__dict_items__": [["label", "target_d64_q128_m131072_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 64], ["K", 64], ["dtype", "bfloat16"], ["seed", 611102], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d128_q768_m98304_k10", {"__dict_items__": [["label", "target_d128_q768_m98304_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 768], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611103], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d768_q64_m65536_k10", {"__dict_items__": [["label", "target_d768_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 611108], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d1024_q32_m32768_k64", {"__dict_items__": [["label", "target_d1024_q32_m32768_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 1024], ["K", 64], ["dtype", "bfloat16"], ["seed", 611110], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d2048_q8_m16384_k10", {"__dict_items__": [["label", "target_d2048_q8_m16384_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 16384], ["D", 2048], ["K", 10], ["dtype", "bfloat16"], ["seed", 611111], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d4096_q4_m8192_k10", {"__dict_items__": [["label", "target_d4096_q4_m8192_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 8192], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 611112], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d4096_q8_m16384_k10", {"__dict_items__": [["label", "target_d4096_q8_m16384_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 16384], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 611113], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["residual0701_d256_q64_m262144_k64", {"__dict_items__": [["label", "residual0701_d256_q64_m262144_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 262144], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 613001], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}]]}'))
TARGET_SHAPES: list[dict[str, Any]] = [_SHAPES_BY_LABEL[TARGET_SHAPE]]
ROUTE = 'residual_full198_d256_k64_fused_hier8x64_ownership_e92c_tcgen05'
ENTRYPOINT = 'loom.examples.weave.knn_search_residual_full198_d256_k64_hierarchical_ownership_0707_e92c_v1:launch_for_eval'
INCUMBENT_ENTRYPOINT = 'loom.examples.weave.knn_search_extendedk_lowd_d256_dispatch0610_r3_extk_v1:launch_for_eval'
SOURCE_PATH = 'loom/examples/weave/knn_search_residual_full198_d256_k64_hierarchical_ownership_0707_e92c_v1.py'
GUARD = 'B == 1 and Q == 128 and M == 131072 and D == 256 and K == 64 and dtype == bfloat16 and not self_search and not force_fallback and arch in {sm_100a,sm_103a}'
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
_KERNELS: dict[str, Any] = {}
knn_search_residual_full198_d256_k64_fused_hier8x64_e92c_v1 = _decode_capture(_json_loads('{"__ir__": "knn_search_residual_full198_d256_k64_fused_hier8x64_e92c_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 4096, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
group_final_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_residual_full198_d256_k64_fused_hier8x64_e92c_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 4096, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_residual_full198_d256_k64_fused_hier8x64_e92c_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 4096, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))

def _compile_kernels() -> dict[str, Any]:
    return _decode_capture(_json_loads('{"group_final": {"__kernel__": "dispatch_kernel_0494"}, "partial": {"__kernel__": "dispatch_kernel_0493"}}'))

def _use_target(inputs: dict[str, Any]) -> bool:
    return int(inputs['B']) == 1 and int(inputs['Q']) == 128 and (int(inputs['M']) == 131072) and (int(inputs['D']) == D_STATIC) and (int(inputs['K']) == K_MAX) and (str(inputs['queries'].dtype) == 'torch.bfloat16') and (not bool(inputs.get('self_search', False))) and (not bool(inputs.get('force_fallback', False))) and producer._tcgen05_capable_arch()

def _launch_target(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _KERNELS:
        _KERNELS.update(_compile_kernels())
    bsz = int(inputs['B'])
    q_rows = int(inputs['Q'])
    m_rows = int(inputs['M'])
    k = int(inputs['K'])
    num_q_tiles = math.ceil(q_rows / BLOCK_Q)
    total_m_tiles = math.ceil(m_rows / BLOCK_M)
    tiles_per_split = math.ceil(total_m_tiles / SPLIT_M)
    partial_distances, partial_indices = producer._scratch(inputs, PARTIAL_LISTS, num_q_tiles)
    _KERNELS['partial'].launch(grid=(bsz * num_q_tiles * SPLIT_M, 1, 1), block=(THREADS, 1, 1), args=[inputs['queries'], inputs['database'], partial_distances, partial_indices, bsz, q_rows, m_rows, SPLIT_M, num_q_tiles, total_m_tiles, tiles_per_split], shared_mem=MMA_SMEM_BYTES)
    _KERNELS['group_final'].launch(grid=(bsz * q_rows, 1, 1), block=(THREADS, 1, 1), args=[partial_distances, partial_indices, inputs['out_distances'], inputs['out_indices'], bsz, q_rows, k, num_q_tiles], shared_mem=FUSED_SMEM_BYTES)
    return {'distances': inputs['out_distances'], 'indices': inputs['out_indices']}

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_target(inputs):
        return ROUTE
    return incumbent.selected_route(inputs)

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    if not _use_target(inputs):
        return incumbent.route_info(inputs)
    return {'route': ROUTE, 'selected_route': ROUTE, 'selected_entrypoint': ENTRYPOINT, 'parent_entrypoint': INCUMBENT_ENTRYPOINT, 'parent_route': incumbent.ROUTE_D256_TCGEN05, 'route_kind': 'specialized', 'route_source': 'bucket-specific-seed', 'classification': 'seed-produced', 'coverage_class': 'residual_full198_d256_k64_hierarchical_ownership', 'production_policy': 'weave_only', 'external_fallback': None, 'arch_requirement': 'sm_100a', 'guard_id': 'd256_k64_hierarchical_ownership_0707_e92c', 'selected_guard': GUARD, 'forced_fallback': False, 'split_m': SPLIT_M, 'tiles_per_split': 8, 'partial_list_count': PARTIAL_LISTS, 'hierarchical_groups': HIERARCHICAL_GROUPS, 'lists_per_group': LISTS_PER_GROUP, 'producer_grid': [SPLIT_M, 1, 1], 'group_final_grid': [128, 1, 1], 'producer_to_consumer_abi': '[B,q_tile,split*2+cohort,q_local,K64] f32/i32 sorted top64', 'consumer_layout': 'eight warp-owned 64-list groups/CTA, CTA-SMEM group handoff, warp0 final eight-way merge', 'group_scratch': 'cta_shared_only', 'final_launch': 'fused_into_group_consumer', 'caller_owned_outputs': True, 'padding_tag': 'none', 'uses_materialized_padding': False, 'uses_kernel_padding': False, 'padding_overhead_timed': False, 'workspace_reuse': 'cached exact-shape partial scratch; producer overwrites all lists'}

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_target(inputs):
        return _launch_target(inputs)
    return incumbent.launch_for_eval(inputs)

def incumbent_launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return incumbent.launch_for_eval(inputs)
