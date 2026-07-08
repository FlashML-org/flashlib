"""Residual-Full198 five-row exact-K64 capability portfolio.

Minimum target architecture: sm_100a.  The two rows that currently reach the
scalar-capacity fallback are structurally reset onto existing exact-shape
tcgen05 scan-to-top64 paths: D64 uses a split-512 two-tile producer followed by
the exact hierarchical K64 merge, while D1024 uses a direct-stride producer
followed by the exact eight-group K64 merge.  The three specialized near-floor
rows retain their exact incumbent producer/merge paths unchanged.

This module is an additive capability portfolio, not a production dispatcher.
It accepts only the five assigned shapes, keeps every scan and top64 consumer
on the Weave path, and writes FP32 distances and INT32 indices directly into
the caller-owned output tensors.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from collections.abc import Callable
from typing import Any
from .. import _dispatch_runtime as contract
from . import knn_search_d64_q128_m131072_k64_0623_e157_v1 as d64
from . import knn_search_d1024_q32_m32768_k64_0623_f561_hiermerge8_v2 as d1024
from . import knn_search_extendedk_lowd_d256_dispatch0610_r3_extk_v1 as d256
from . import knn_search_k64_q4096split79_localprefix6_certfallback_0615_245d_v1 as q4096
from . import knn_search_residual0705_q64_warp_distributed_state_04b4_v1 as residual
from . import knn_search_scalar_capacity_0611_r22_4e96_v1 as scalar_capacity
ENTRYPOINT = 'loom.examples.weave.knn_search_residual_full198_large_k_capacity_capability_0707_0334_v1:launch_for_eval'
SOURCE_PATH = 'loom/examples/weave/knn_search_residual_full198_large_k_capacity_capability_0707_0334_v1.py'
INCUMBENT_ENTRYPOINT = 'loom.examples.weave.knn_search_residual_full198_large_k_capacity_capability_0707_0334_v1:incumbent_launch_for_eval'
TARGET_SHAPE = 'target_d64_q128_m131072_k64'
TARGET_LABELS: tuple[str, ...] = ('target_d64_q128_m131072_k64', 'target_d1024_q32_m32768_k64', 'glm5_rag_q128_m131072_d256_k64', 'ksweep_q4096_m20000_d128_k64', 'residual0701_d256_q64_m262144_k64')
_SHAPES_BY_LABEL = _decode_capture(_json_loads('{"__dict_items__": [["ksweep_q4096_m20000_d128_k64", {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610313], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["glm5_rag_q128_m131072_d256_k64", {"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 610406], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d33_q128_m131072_k10", {"__dict_items__": [["label", "blind_0622_dyn_d33_q128_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 33], ["K", 10], ["dtype", "bfloat16"], ["seed", 611005], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d80_q256_m65536_k10", {"__dict_items__": [["label", "blind_0622_dyn_d80_q256_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 65536], ["D", 80], ["K", 10], ["dtype", "bfloat16"], ["seed", 611006], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d160_q64_m131072_k10", {"__dict_items__": [["label", "blind_0622_dyn_d160_q64_m131072_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 131072], ["D", 160], ["K", 10], ["dtype", "bfloat16"], ["seed", 611007], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_dyn_d640_q32_m32768_k10", {"__dict_items__": [["label", "blind_0622_dyn_d640_q32_m32768_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 640], ["K", 10], ["dtype", "bfloat16"], ["seed", 611008], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_b2_q96_m196608_d128_k10", {"__dict_items__": [["label", "blind_0622_b2_q96_m196608_d128_k10"], ["params", {"__dict_items__": [["B", 2], ["Q", 96], ["M", 196608], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611009], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["blind_0622_tail_q1025_m65537_d128_k10", {"__dict_items__": [["label", "blind_0622_tail_q1025_m65537_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1025], ["M", 65537], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611012], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d64_q128_m131072_k64", {"__dict_items__": [["label", "target_d64_q128_m131072_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 64], ["K", 64], ["dtype", "bfloat16"], ["seed", 611102], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d128_q768_m98304_k10", {"__dict_items__": [["label", "target_d128_q768_m98304_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 768], ["M", 98304], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 611103], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d768_q64_m65536_k10", {"__dict_items__": [["label", "target_d768_q64_m65536_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 65536], ["D", 768], ["K", 10], ["dtype", "bfloat16"], ["seed", 611108], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d1024_q32_m32768_k64", {"__dict_items__": [["label", "target_d1024_q32_m32768_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 32], ["M", 32768], ["D", 1024], ["K", 64], ["dtype", "bfloat16"], ["seed", 611110], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d2048_q8_m16384_k10", {"__dict_items__": [["label", "target_d2048_q8_m16384_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 16384], ["D", 2048], ["K", 10], ["dtype", "bfloat16"], ["seed", 611111], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d4096_q4_m8192_k10", {"__dict_items__": [["label", "target_d4096_q4_m8192_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4], ["M", 8192], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 611112], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["target_d4096_q8_m16384_k10", {"__dict_items__": [["label", "target_d4096_q8_m16384_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 16384], ["D", 4096], ["K", 10], ["dtype", "bfloat16"], ["seed", 611113], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}], ["residual0701_d256_q64_m262144_k64", {"__dict_items__": [["label", "residual0701_d256_q64_m262144_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 64], ["M", 262144], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 613001], ["self_search", false], ["min_recall", 0.999], ["check_correctness", true], ["benchmark", true]]}]]}]]}'))
TARGET_SHAPES: list[dict[str, Any]] = [_SHAPES_BY_LABEL[label] for label in TARGET_LABELS]
_D64_KEY = (1, 128, 131072, 64, 64, False)
_D1024_KEY = (1, 32, 32768, 1024, 64, False)
_D256_Q128_KEY = (1, 128, 131072, 256, 64, False)
_D128_Q4096_KEY = (1, 4096, 20000, 128, 64, False)
_D256_Q64_KEY = (1, 64, 262144, 256, 64, False)
_LABEL_BY_KEY = {_D64_KEY: TARGET_LABELS[0], _D1024_KEY: TARGET_LABELS[1], _D256_Q128_KEY: TARGET_LABELS[2], _D128_Q4096_KEY: TARGET_LABELS[3], _D256_Q64_KEY: TARGET_LABELS[4]}
_CANDIDATE_ROUTE_BY_KEY = {_D64_KEY: 'target_d64_q128_m131072_k64_split512_d64_tcgen05', _D1024_KEY: d1024.ROUTE_D1024_Q32_K64_HIERMERGE8, _D256_Q128_KEY: d256.ROUTE_D256_TCGEN05, _D128_Q4096_KEY: q4096.ROUTE_Q4096_M20000_K64_PREFIX6CERT, _D256_Q64_KEY: residual.ROUTE}
_INCUMBENT_ROUTE_BY_KEY = {_D64_KEY: 'afe6_dynamic_d_scalar_capacity', _D1024_KEY: 'afe6_dynamic_d_scalar_capacity', _D256_Q128_KEY: d256.ROUTE_D256_TCGEN05, _D128_Q4096_KEY: q4096.ROUTE_Q4096_M20000_K64_PREFIX6CERT, _D256_Q64_KEY: residual.ROUTE}
_CANDIDATE_UNDERLYING_ENTRYPOINT_BY_KEY = {_D64_KEY: 'loom.examples.weave.knn_search_d64_q128_m131072_k64_0623_e157_v1:launch_for_eval', _D1024_KEY: 'loom.examples.weave.knn_search_d1024_q32_m32768_k64_0623_f561_hiermerge8_v2:launch_for_eval', _D256_Q128_KEY: 'loom.examples.weave.knn_search_extendedk_lowd_d256_dispatch0610_r3_extk_v1:launch_for_eval', _D128_Q4096_KEY: 'loom.examples.weave.knn_search_k64_q4096split79_localprefix6_certfallback_0615_245d_v1:launch_for_eval', _D256_Q64_KEY: 'loom.examples.weave.knn_search_residual0705_q64_warp_distributed_state_04b4_v1:launch_for_eval'}
_INCUMBENT_UNDERLYING_ENTRYPOINT_BY_KEY = {_D64_KEY: 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_scalar_capacity_for_eval', _D1024_KEY: 'loom.examples.weave.knn_search_scalar_capacity_0611_r22_4e96_v1:launch_scalar_capacity_for_eval', _D256_Q128_KEY: _CANDIDATE_UNDERLYING_ENTRYPOINT_BY_KEY[_D256_Q128_KEY], _D128_Q4096_KEY: _CANDIDATE_UNDERLYING_ENTRYPOINT_BY_KEY[_D128_Q4096_KEY], _D256_Q64_KEY: _CANDIDATE_UNDERLYING_ENTRYPOINT_BY_KEY[_D256_Q64_KEY]}
STRUCTURAL_DELTAS: dict[str, str] = {TARGET_LABELS[0]: 'scalar D-loop plus scalar K64 insertion -> D64-specialized split512 two-tile tcgen05 producer -> exact 32-group K64 merge -> caller outputs', TARGET_LABELS[1]: 'scalar D-loop plus scalar K64 insertion -> direct-stride D1024 tcgen05 producer -> exact hiermerge8 K64 consumer -> caller outputs', TARGET_LABELS[2]: 'retained exact round36 D256 tcgen05 K64 incumbent', TARGET_LABELS[3]: 'retained exact round20 Q4096 split79 K64 incumbent', TARGET_LABELS[4]: 'retained exact residual0701 warp-distributed K64 incumbent'}
CAPABILITY_SHAPE_REGISTRY = _decode_capture(_json_loads('{"__tuple__": [{"__dict_items__": [["shape_key", "target_d64_q128_m131072_k64"], ["guard", "dtype == bfloat16 and exact (B,Q,M,D,K,self_search) == (1, 128, 131072, 64, 64, False) and force_fallback == false and arch in {sm_100a,sm_103a}"], ["candidate_route", "target_d64_q128_m131072_k64_split512_d64_tcgen05"], ["incumbent_route", "afe6_dynamic_d_scalar_capacity"], ["entrypoint", "loom.examples.weave.knn_search_residual_full198_large_k_capacity_capability_0707_0334_v1:launch_for_eval"], ["underlying_entrypoint", "loom.examples.weave.knn_search_d64_q128_m131072_k64_0623_e157_v1:launch_for_eval"], ["structural_delta", "scalar D-loop plus scalar K64 insertion -> D64-specialized split512 two-tile tcgen05 producer -> exact 32-group K64 merge -> caller outputs"], ["retained_incumbent", false]]}, {"__dict_items__": [["shape_key", "target_d1024_q32_m32768_k64"], ["guard", "dtype == bfloat16 and exact (B,Q,M,D,K,self_search) == (1, 32, 32768, 1024, 64, False) and force_fallback == false and arch in {sm_100a,sm_103a}"], ["candidate_route", "f561_v2_d1024_q32_k64_hiermerge8_tcgen05"], ["incumbent_route", "afe6_dynamic_d_scalar_capacity"], ["entrypoint", "loom.examples.weave.knn_search_residual_full198_large_k_capacity_capability_0707_0334_v1:launch_for_eval"], ["underlying_entrypoint", "loom.examples.weave.knn_search_d1024_q32_m32768_k64_0623_f561_hiermerge8_v2:launch_for_eval"], ["structural_delta", "scalar D-loop plus scalar K64 insertion -> direct-stride D1024 tcgen05 producer -> exact hiermerge8 K64 consumer -> caller outputs"], ["retained_incumbent", false]]}, {"__dict_items__": [["shape_key", "glm5_rag_q128_m131072_d256_k64"], ["guard", "dtype == bfloat16 and exact (B,Q,M,D,K,self_search) == (1, 128, 131072, 256, 64, False) and force_fallback == false and arch in {sm_100a,sm_103a}"], ["candidate_route", "round36_d256_k10_k64_tcgen05"], ["incumbent_route", "round36_d256_k10_k64_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_residual_full198_large_k_capacity_capability_0707_0334_v1:launch_for_eval"], ["underlying_entrypoint", "loom.examples.weave.knn_search_extendedk_lowd_d256_dispatch0610_r3_extk_v1:launch_for_eval"], ["structural_delta", "retained exact round36 D256 tcgen05 K64 incumbent"], ["retained_incumbent", true]]}, {"__dict_items__": [["shape_key", "ksweep_q4096_m20000_d128_k64"], ["guard", "dtype == bfloat16 and exact (B,Q,M,D,K,self_search) == (1, 4096, 20000, 128, 64, False) and force_fallback == false and arch in {sm_100a,sm_103a}"], ["candidate_route", "round20_245d_q4096_m20000_k64_prefix6cert"], ["incumbent_route", "round20_245d_q4096_m20000_k64_prefix6cert"], ["entrypoint", "loom.examples.weave.knn_search_residual_full198_large_k_capacity_capability_0707_0334_v1:launch_for_eval"], ["underlying_entrypoint", "loom.examples.weave.knn_search_k64_q4096split79_localprefix6_certfallback_0615_245d_v1:launch_for_eval"], ["structural_delta", "retained exact round20 Q4096 split79 K64 incumbent"], ["retained_incumbent", true]]}, {"__dict_items__": [["shape_key", "residual0701_d256_q64_m262144_k64"], ["guard", "dtype == bfloat16 and exact (B,Q,M,D,K,self_search) == (1, 64, 262144, 256, 64, False) and force_fallback == false and arch in {sm_100a,sm_103a}"], ["candidate_route", "residual0705_q64_warp_distributed_state_04b4_tcgen05"], ["incumbent_route", "residual0705_q64_warp_distributed_state_04b4_tcgen05"], ["entrypoint", "loom.examples.weave.knn_search_residual_full198_large_k_capacity_capability_0707_0334_v1:launch_for_eval"], ["underlying_entrypoint", "loom.examples.weave.knn_search_residual0705_q64_warp_distributed_state_04b4_v1:launch_for_eval"], ["structural_delta", "retained exact residual0701 warp-distributed K64 incumbent"], ["retained_incumbent", true]]}]}'))
d64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d64_q128_m131072_k64_partial_0623_e157_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 57600, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))
d64_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_groupmerge64_kexact_0614_r25_k64thin_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
d64_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q128split512_finalmerge32_kexact_0614_r25_k64thin_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
d1024_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d1024_q32_k64_targetd_partial_67ec_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 92672, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
d1024_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d1024_q32_k64_hiermerge8_group_f561_v2", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
d1024_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d1024_q32_k64_hiermerge8_final_f561_v2", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
d256_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
d256_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
q4096_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix7_partial_0615_245d_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 165120, "constants": [["K_MAX_", 64], ["K_PREFIX_", 7]], "cta_group": 1, "threads": 512}'))
q4096_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_k64_q4096split79_localprefix6_certmerge_0615_245d_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "partial_list_count", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64], ["K_PREFIX_READ_", 6], ["K_STRIDE_", 7]], "cta_group": 1, "threads": 32}'))
residual_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_warp_distributed_state_04b4_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 106240, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
residual_group_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_groupmerge_cce0_v1", "arg_keys": ["partial_distances", "partial_indices", "group_distances", "group_indices", "B", "Q", "K", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 128}'))
residual_final_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q64_pairedowner_finalmerge_cce0_v1", "arg_keys": ["group_distances", "group_indices", "out_distances", "out_indices", "B", "Q", "K"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 256}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_d64_q128_m131072_k64_partial_0623_e157_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 57600, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 512}'))

def _shape_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    return (int(inputs['B']), int(inputs['Q']), int(inputs['M']), int(inputs['D']), int(inputs['K']), bool(inputs.get('self_search', False)))

def _require_owned_key(inputs: dict[str, Any]) -> tuple[int, int, int, int, int, bool]:
    key = _shape_key(inputs)
    if key not in _LABEL_BY_KEY:
        raise ValueError(''.join(['large_k_capacity_capability supports only the five assigned exact K64 shapes; got ', format(repr(key), '')]))
    if bool(inputs.get('force_fallback', False)):
        raise ValueError('large_k_capacity_capability does not permit forced fallback')
    if str(inputs['queries'].dtype) != 'torch.bfloat16':
        raise ValueError(''.join(['large_k_capacity_capability requires BF16 queries/database; got ', format(str(inputs['queries'].dtype), '')]))
    return key

def _candidate_binding(key: tuple[int, int, int, int, int, bool]) -> tuple[Callable[[dict[str, Any]], dict[str, Any]], Callable[[dict[str, Any]], str]]:
    if key == _D64_KEY:
        return (d64.launch_for_eval, d64.selected_route_name)
    if key == _D1024_KEY:
        return (d1024.launch_for_eval, d1024.selected_route_name)
    if key == _D256_Q128_KEY:
        return (d256.launch_for_eval, d256.selected_route_name)
    if key == _D128_Q4096_KEY:
        return (q4096.launch_for_eval, q4096.selected_route_name)
    if key == _D256_Q64_KEY:
        return (residual.launch_for_eval, residual.selected_route)
    raise AssertionError(''.join(['unhandled owned key ', format(repr(key), '')]))

def _incumbent_binding(key: tuple[int, int, int, int, int, bool]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    if key in {_D64_KEY, _D1024_KEY}:
        return scalar_capacity.launch_scalar_capacity_for_eval
    if key == _D256_Q128_KEY:
        return d256.launch_for_eval
    if key == _D128_Q4096_KEY:
        return q4096.launch_for_eval
    if key == _D256_Q64_KEY:
        return residual.launch_for_eval
    raise AssertionError(''.join(['unhandled owned key ', format(repr(key), '')]))

def selected_route(inputs: dict[str, Any]) -> str:
    key = _require_owned_key(inputs)
    _, selector = _candidate_binding(key)
    actual = selector(inputs)
    expected = _CANDIDATE_ROUTE_BY_KEY[key]
    if actual != expected:
        raise RuntimeError(''.join(['candidate guard for ', format(_LABEL_BY_KEY[key], ''), ' selected ', format(repr(actual), ''), '; expected ', format(repr(expected), '')]))
    return actual

def selected_route_name(inputs: dict[str, Any]) -> str:
    return selected_route(inputs)

def route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    key = _require_owned_key(inputs)
    route = selected_route(inputs)
    label = _LABEL_BY_KEY[key]
    return {'route': route, 'selected_route': route, 'selected_entrypoint': ENTRYPOINT, 'underlying_entrypoint': _CANDIDATE_UNDERLYING_ENTRYPOINT_BY_KEY[key], 'route_kind': 'specialized', 'route_source': 'residual-full198-large-k-capability-0707-0334', 'coverage_class': 'exact_k64_scan_to_top64_capability', 'classification': 'seed-produced' if route != _INCUMBENT_ROUTE_BY_KEY[key] else 'incumbent-retained', 'coverage_only': False, 'production_policy': 'weave_only', 'external_fallback': None, 'missing_weave_route': False, 'guard_id': ''.join(['large_k_capacity_capability_0707_0334_', format(label, '')]), 'selected_guard': next((row['guard'] for row in CAPABILITY_SHAPE_REGISTRY if row['shape_key'] == label)), 'parent_route': _INCUMBENT_ROUTE_BY_KEY[key], 'structural_delta': STRUCTURAL_DELTAS[label], 'retained_incumbent': route == _INCUMBENT_ROUTE_BY_KEY[key], 'workspace_reuse': 'exact-shape producer/group/final scratch caches', 'timed_contract_regions': ['producer', 'partial_top64', 'merge', 'caller_output_writes']}

def incumbent_route_info(inputs: dict[str, Any]) -> dict[str, Any]:
    key = _require_owned_key(inputs)
    return {'route': _INCUMBENT_ROUTE_BY_KEY[key], 'selected_route': _INCUMBENT_ROUTE_BY_KEY[key], 'selected_entrypoint': INCUMBENT_ENTRYPOINT, 'underlying_entrypoint': _INCUMBENT_UNDERLYING_ENTRYPOINT_BY_KEY[key], 'route_kind': 'specialized' if key not in {_D64_KEY, _D1024_KEY} else 'fallback', 'route_source': 'exact-retained-incumbent-bank', 'shape_key': _LABEL_BY_KEY[key]}

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    key = _require_owned_key(inputs)
    kernel_fn, _ = _candidate_binding(key)
    selected_route(inputs)
    return kernel_fn(inputs)

def incumbent_launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    key = _require_owned_key(inputs)
    return _incumbent_binding(key)(inputs)

def route_trace_entry(label: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {'label': label, 'shape_key': label, 'B': int(inputs['B']), 'Q': int(inputs['Q']), 'M': int(inputs['M']), 'D': int(inputs['D']), 'K': int(inputs['K']), 'self_search': bool(inputs.get('self_search', False)), **route_info(inputs)}

def compile_and_launch_capability(*, benchmark: bool=True) -> dict[str, Any]:
    from .. import _dispatch_runtime as capability_eval
    return capability_eval.evaluate(launch_for_eval, incumbent_launch_for_eval, shapes=TARGET_SHAPES, benchmark=benchmark)
