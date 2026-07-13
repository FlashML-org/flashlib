"""Round-47 registered Q4096 low-K K1 + K8 routes for exact BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05 routes. This additive
dispatcher layers the round-52 ``B=1,Q=4096,M=20000,D=128,K=8`` output-8 route
on top of the round-46 K1 min-pair registered dispatcher. K1, K2, K5, K10,
K64, low-D, D256, and every guard miss delegate to round 46 unchanged.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_highq_midm_k8_out8_split4_0613_r52_48e9_v1 as k8_route
from . import knn_search_q4096_lowk_k1partial_minpair_registered_0613_r46_48e9_lowk_k1top1_v1 as parent
THREADS = parent.THREADS
MERGE_THREADS = parent.MERGE_THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
Q4096_ROWS = parent.Q4096_ROWS
Q4096_LOWK_M = parent.Q4096_LOWK_M
Q4096_LOWK_K1_SPLIT_M = parent.Q4096_LOWK_K1_SPLIT_M
Q4096_LOWK_K8_SPLIT_M = k8_route.Q4096_LOWK_K8_SPLIT_M
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
Q4096_K1_SHAPES = parent.Q4096_K1_SHAPES
Q4096_K2_SHAPES = parent.Q4096_K2_SHAPES
Q4096_K5_SHAPES = parent.Q4096_K5_SHAPES
Q4096_K8_SHAPE = _decode_capture(_json_loads('{"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k8"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 610315], ["self_search", false], ["min_recall", 0.999]]}]]}'))
Q4096_K8_SHAPES: list[dict[str, Any]] = [Q4096_K8_SHAPE]
Q4096_LOWK_SHAPES = [*parent.Q4096_LOWK_SHAPES, *Q4096_K8_SHAPES]
DEFAULT_REGISTRY_CORRECTNESS_LABELS = parent.DEFAULT_REGISTRY_CORRECTNESS_LABELS
DEFAULT_REGISTRY_PERFORMANCE_LABELS = parent.DEFAULT_REGISTRY_PERFORMANCE_LABELS
DEFAULT_REGISTRY_CORRECTNESS_SHAPES = parent.DEFAULT_REGISTRY_CORRECTNESS_SHAPES
DEFAULT_REGISTRY_PERFORMANCE_SHAPES = [*parent.DEFAULT_REGISTRY_PERFORMANCE_SHAPES, *Q4096_K8_SHAPES]
ROUND47_PRESERVE_SHAPES = [*parent.ROUND46_PRESERVE_SHAPES, *Q4096_K8_SHAPES]
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'registered_d128_q4096_lowk_k8_stride10_out8_split4', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 8 and tcgen05', 'route': 'round47_registered_q4096_lowk_k8_stride10_out8_split4'}, *parent.SHAPE_DISPATCH_REGISTRY)

def _use_q4096_k8(inputs: dict[str, Any]) -> bool:
    return k8_route._use_q4096_k8(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_k8(inputs):
        return 'round47_registered_q4096_lowk_k8_stride10_out8_split4'
    return parent.selected_route(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    if _use_q4096_k8(inputs):
        return k8_route._launch_q4096_k8_split4(inputs)
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_q4096_k8(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=Q4096_K8_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_q4096_lowk(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=Q4096_LOWK_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_registered_default(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    if benchmark:
        shapes = DEFAULT_REGISTRY_PERFORMANCE_SHAPES
    else:
        shapes = [*DEFAULT_REGISTRY_CORRECTNESS_SHAPES, *parent.Q4096_LOWK_SHAPES, *Q4096_K8_SHAPES]
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
