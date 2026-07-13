"""Round-50 registered high-Q K5-capacity dispatcher for BF16 kNN.

Minimum target architecture: sm_100a for the tcgen05 high-Q MMA routes. This
dispatcher promotes the round-49 ``B=1,Q=4096,M=20000,D=128,K=5`` five-slot
producer route into the registered/default kNN path while preserving the
round-49 dispatcher behavior for every guard miss.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_q4096_lowk_k5partial_dispatch_0613_r49_48e9_v1 as promoted
THREADS = promoted.THREADS
MERGE_THREADS = promoted.MERGE_THREADS
BLOCK_Q = promoted.BLOCK_Q
BLOCK_M = promoted.BLOCK_M
D_STATIC = promoted.D_STATIC
K_PARTIAL_MAX = promoted.K_PARTIAL_MAX
Q4096_ROWS = promoted.Q4096_ROWS
Q4096_LOWK_M = promoted.Q4096_LOWK_M
Q4096_LOWK_K5_SPLIT_M = promoted.Q4096_LOWK_K5_SPLIT_M
partial_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_0613_r44_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k1_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k1partial_merge_0613_r44_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [], "cta_group": 1, "threads": 32}'))
partial_k2_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k2partial_0613_r45_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k2_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k2partial_split9_merge_0613_r46_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 2], ["K_OUT_MAX_", 2]], "cta_group": 1, "threads": 32}'))
partial_k5_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
merge_k5_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_merge_0613_r49_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 5], ["K_OUT_MAX_", 5]], "cta_group": 1, "threads": 32}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5_stride10_merge_0613_r48_48e9_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_STRIDE_", 10], ["K_OUT_MAX_", 5]], "cta_group": 1, "threads": 32}'))
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_q4096_lowk_k5partial_0613_r49_48e9_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [], "cta_group": 1, "threads": 640}'))
Q4096_K5_LABELS = promoted.Q4096_K5_LABELS
Q4096_K5_SHAPES = promoted.Q4096_K5_SHAPES
HIGHQ_MIDM_REGISTERED_LABELS = promoted.ROUND49_MEASURE_LABELS
HIGHQ_MIDM_REGISTERED_SHAPES = promoted.ROUND49_MEASURE_SHAPES
ROUND50_PRESERVE_SHAPES = promoted.ROUND49_PRESERVE_SHAPES
DEFAULT_REGISTRY_CORRECTNESS_LABELS: tuple[str, ...] = ('flashml_self_b1_q256_m256_d128_k5', 'rag_q1_m131072_d128_k10', 'rag_q128_m131072_d128_k10', 'rag_q4096_m20000_d128_k10', 'ivf_like_q8_m10_d32_k10', 'ivf_like_q8_m20_d48_k10', 'dbscan_lowd_self_q1500_m1500_d2_k32', 'dbscan_lowd_self_q1500_m1500_d2_k64', 'glm5_rag_q128_m131072_d256_k10', 'glm5_rag_q128_m131072_d256_k64', 'ksweep_q4096_m20000_d128_k64')
DEFAULT_REGISTRY_PERFORMANCE_LABELS: tuple[str, ...] = ('rag_q128_m131072_d128_k10', 'rag_q4096_m20000_d128_k10')
DEFAULT_REGISTRY_CORRECTNESS_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "flashml_self_b1_q256_m256_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 256], ["M", 256], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 0], ["self_search", true], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.99]]}]]}, {"__dict_items__": [["label", "rag_q1_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 1], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 1], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "rag_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 2], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 3], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ivf_like_q8_m10_d32_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 10], ["D", 32], ["K", 10], ["dtype", "bfloat16"], ["seed", 610403], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ivf_like_q8_m20_d48_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 20], ["D", 48], ["K", 10], ["dtype", "bfloat16"], ["seed", 610404], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "dbscan_lowd_self_q1500_m1500_d2_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 1500], ["M", 1500], ["D", 2], ["K", 32], ["dtype", "bfloat16"], ["seed", 610405], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dbscan_lowd_self_q1500_m1500_d2_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 1500], ["M", 1500], ["D", 2], ["K", 64], ["dtype", "bfloat16"], ["seed", 610407], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 610402], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 610406], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610313], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
DEFAULT_REGISTRY_PERFORMANCE_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "rag_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 2], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_q4096_m20000_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 3], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 610314], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'registered_d128_q4096_lowk_k5partial_split4', 'guard': 'B == 1 and Q == 4096 and M == 20000 and D == 128 and K == 5 and tcgen05', 'route': 'round50_registered_q4096_lowk_k5partial_split4'}, *promoted.SHAPE_DISPATCH_REGISTRY)

def _use_q4096_k5(inputs: dict[str, Any]) -> bool:
    return promoted._use_q4096_k5(inputs)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q4096_k5(inputs):
        return 'round50_registered_q4096_lowk_k5partial_split4'
    return promoted.selected_route(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return promoted.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_registered_default(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    if benchmark:
        shapes = DEFAULT_REGISTRY_PERFORMANCE_SHAPES
    else:
        shapes = [*DEFAULT_REGISTRY_CORRECTNESS_SHAPES, *Q4096_K5_SHAPES]
    result = evaluate(launch_for_eval, shapes=shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_highq_midm_registered(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=HIGHQ_MIDM_REGISTERED_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
