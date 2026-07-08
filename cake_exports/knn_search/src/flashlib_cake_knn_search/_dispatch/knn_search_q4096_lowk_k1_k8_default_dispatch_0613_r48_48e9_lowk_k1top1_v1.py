"""Default dispatcher promotion for exact BF16 kNN low-K routes.

Minimum target architecture: sm_100a for the tcgen05 low-K routes. This
dispatcher-policy wrapper routes the round-54
``B=1,Q=4096,D=128,K=8,M in {16384,32768}`` M-bucket shapes before preserving
the promoted ``M=20000`` K8 route and every inherited K1/K2/K5/K10/K64
guard-miss route.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from . import knn_search_highq_midm_k8_mbucket_split4_0613_r54_48e9_v1 as k8_mbucket, knn_search_q4096_lowk_k1_k8_registered_0613_r47_48e9_lowk_k1top1_v1 as promoted
THREADS = promoted.THREADS
MERGE_THREADS = promoted.MERGE_THREADS
BLOCK_Q = promoted.BLOCK_Q
BLOCK_M = promoted.BLOCK_M
D_STATIC = promoted.D_STATIC
Q4096_ROWS = promoted.Q4096_ROWS
Q4096_LOWK_M = promoted.Q4096_LOWK_M
Q4096_LOWK_K1_SPLIT_M = promoted.Q4096_LOWK_K1_SPLIT_M
Q4096_LOWK_K8_SPLIT_M = promoted.Q4096_LOWK_K8_SPLIT_M
Q4096_K8_M_BUCKETS = k8_mbucket.Q4096_K8_M_BUCKETS
Q4096_LOWK_K8_MBUCKET_SPLIT_M = k8_mbucket.Q4096_LOWK_K8_MBUCKET_SPLIT_M
partial_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_minpair_0613_r46_48e9_lowk_k1top1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_minpair_merge_0613_r46_48e9_lowk_k1top1_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
partial_k2_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k2partial_0613_r45_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k2_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k2partial_split9_merge_0613_r46_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 2], ["K_OUT_MAX_", 2]], "cta_group": 1, "threads": 32}'))
partial_k5_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k5_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_split9_merge_0613_r51_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 5], ["K_OUT_MAX_", 5]], "cta_group": 1, "threads": 32}'))
partial_k8_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
merge_k8_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k8_stride10_out8_merge_0613_r52_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 10], ["K_OUT_MAX_", 8]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_minpair_0613_r46_48e9_lowk_k1top1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_minpair_0613_r46_48e9_lowk_k1top1_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
Q4096_K1_SHAPES = promoted.Q4096_K1_SHAPES
Q4096_K2_SHAPES = promoted.Q4096_K2_SHAPES
Q4096_K5_SHAPES = promoted.Q4096_K5_SHAPES
Q4096_K8_SHAPES = promoted.Q4096_K8_SHAPES
Q4096_K8_MBUCKET_SHAPES = k8_mbucket.Q4096_K8_MBUCKET_SHAPES
Q4096_LOWK_SHAPES = [*promoted.Q4096_LOWK_SHAPES, *Q4096_K8_MBUCKET_SHAPES]
DEFAULT_REGISTRY_CORRECTNESS_LABELS = promoted.DEFAULT_REGISTRY_CORRECTNESS_LABELS
DEFAULT_REGISTRY_PERFORMANCE_LABELS = promoted.DEFAULT_REGISTRY_PERFORMANCE_LABELS
DEFAULT_REGISTRY_CORRECTNESS_SHAPES = promoted.DEFAULT_REGISTRY_CORRECTNESS_SHAPES
DEFAULT_REGISTRY_PERFORMANCE_SHAPES = [*promoted.DEFAULT_REGISTRY_PERFORMANCE_SHAPES, *Q4096_K8_MBUCKET_SHAPES]
ROUND48_PRESERVE_SHAPES = [*promoted.ROUND47_PRESERVE_SHAPES, *Q4096_K8_MBUCKET_SHAPES]
SHAPE_DISPATCH_REGISTRY = (*k8_mbucket.SHAPE_DISPATCH_REGISTRY[:2], *promoted.SHAPE_DISPATCH_REGISTRY)

def _use_q4096_k8_mbucket(inputs: dict[str, Any]) -> bool:
    return k8_mbucket._use_q4096_k8_mbucket(inputs)

def _use_q4096_k8(inputs: dict[str, Any]) -> bool:
    return promoted._use_q4096_k8(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_k8_mbucket(inputs):
        return k8_mbucket.selected_route(inputs)
    return promoted.selected_route(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k8_mbucket(inputs):
        return k8_mbucket._launch_q4096_k8_mbucket_split4(inputs)
    return promoted.launch_for_eval(inputs)

def knn_search_compile_and_launch_promoted_default(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    if benchmark:
        shapes = DEFAULT_REGISTRY_PERFORMANCE_SHAPES
    else:
        shapes = [*DEFAULT_REGISTRY_CORRECTNESS_SHAPES, *Q4096_LOWK_SHAPES]
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_registered_default(*, benchmark: bool=True) -> dict[str, Any]:
    return knn_search_compile_and_launch_promoted_default(benchmark=benchmark)
