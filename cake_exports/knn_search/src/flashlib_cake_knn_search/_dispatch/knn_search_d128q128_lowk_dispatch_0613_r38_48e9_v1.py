"""Round-38 Q128 low-K dispatcher over the r37 kNN routes.

Minimum target architecture: sm_100a for the D128/Q128 and D256 tcgen05
routes. This dispatcher-phase wrapper preserves the round-37 launch behavior
and makes the Q128/D128 large-M low-K labels explicit in the route registry and
eval shape sets.
"""
from __future__ import annotations
from json import loads as _json_loads
from .._dispatch_runtime import _capture_cuTensorMapEncodeTiled, _decode_capture, _import_dispatch_module, _ir_proxy
from typing import Any
from .._dispatch_runtime import select_named_shapes
from . import knn_search_d128q128_lowd_d256_dispatch_0612_r37_48e9_v1 as parent
THREADS = parent.THREADS
BLOCK_Q = parent.BLOCK_Q
BLOCK_M = parent.BLOCK_M
D_STATIC = parent.D_STATIC
K_MAX = parent.K_MAX
ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 108800, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 0], ["FULL_M_TILES", 0]], "cta_group": 1, "threads": 640}'))
parent_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
d256_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 10], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
d256_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 10]], "cta_group": 1, "threads": 32}'))
k64_partial_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_partial_v1", "arg_keys": ["queries", "database", "partial_distances", "partial_indices", "B", "Q", "M", "split_m", "num_q_tiles", "total_m_tiles", "tiles_per_split"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 143104, "constants": [["K_MAX_", 64], ["EXPOSE_COL_COHORTS", 1]], "cta_group": 1, "threads": 256}'))
k64_merge_ir = _decode_capture(_json_loads('{"__ir__": "knn_search_mma_split_merge_v1", "arg_keys": ["partial_distances", "partial_indices", "out_distances", "out_indices", "B", "Q", "K", "split_m", "num_q_tiles"], "cluster_dims": [1, 1, 1], "computed_smem_bytes": 0, "constants": [["K_MAX_", 64]], "cta_group": 1, "threads": 32}'))
Q128_LOWK_VALUES: tuple[int, ...] = (1, 2, 5, 8, 10)
Q128_LOWK_SHAPE_LABELS: tuple[str, ...] = ('ksweep_q128_m131072_d128_k1', 'ksweep_q128_m131072_d128_k2', 'ksweep_q128_m131072_d128_k5', 'ksweep_q128_m131072_d128_k8', 'ksweep_q128_m131072_d128_k10')
D256_REGISTERED_SHAPE_LABELS: tuple[str, ...] = ('glm5_rag_q128_m131072_d256_k10', 'glm5_rag_q128_m131072_d256_k64')
ROUND38_GUARDMISS_PRESERVE_LABELS: tuple[str, ...] = ('ivf_like_q8_m10_d32_k10', 'ivf_like_q8_m20_d48_k10', 'dbscan_lowd_self_q1500_m1500_d2_k32', 'dbscan_lowd_self_q1500_m1500_d2_k64', 'ksweep_q4096_m20000_d128_k64')
Q128_LOWK_DISPATCH_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "ksweep_q128_m131072_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610301], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k2"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 2], ["dtype", "bfloat16"], ["seed", 610302], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 610303], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k8"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 610304], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610305], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
D256_REGISTERED_DISPATCH_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 610402], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 610406], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
ROUND38_DISPATCHER_SHAPES = [*parent.D128_Q128_DISPATCH_SHAPES, *Q128_LOWK_DISPATCH_SHAPES, *D256_REGISTERED_DISPATCH_SHAPES]
ROUND38_PRESERVE_SHAPES = _decode_capture(_json_loads('[{"__dict_items__": [["label", "dispatch_q128_m8192_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 8192], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610201], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m16384_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 16384], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610202], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m32768_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 32768], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610203], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m65536_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 65536], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610204], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 2], ["self_search", false], ["check_correctness", true], ["benchmark", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "rag_batch_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610109], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dispatch_q128_m262144_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 262144], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610205], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k1"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 1], ["dtype", "bfloat16"], ["seed", 610301], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k2"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 2], ["dtype", "bfloat16"], ["seed", 610302], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k5"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 5], ["dtype", "bfloat16"], ["seed", 610303], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k8"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 8], ["dtype", "bfloat16"], ["seed", 610304], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q128_m131072_d128_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 128], ["K", 10], ["dtype", "bfloat16"], ["seed", 610305], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 10], ["dtype", "bfloat16"], ["seed", 610402], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "glm5_rag_q128_m131072_d256_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 128], ["M", 131072], ["D", 256], ["K", 64], ["dtype", "bfloat16"], ["seed", 610406], ["self_search", false], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ivf_like_q8_m10_d32_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 10], ["D", 32], ["K", 10], ["dtype", "bfloat16"], ["seed", 610403], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "ivf_like_q8_m20_d48_k10"], ["params", {"__dict_items__": [["B", 1], ["Q", 8], ["M", 20], ["D", 48], ["K", 10], ["dtype", "bfloat16"], ["seed", 610404], ["self_search", false], ["min_recall", 1.0]]}]]}, {"__dict_items__": [["label", "dbscan_lowd_self_q1500_m1500_d2_k32"], ["params", {"__dict_items__": [["B", 1], ["Q", 1500], ["M", 1500], ["D", 2], ["K", 32], ["dtype", "bfloat16"], ["seed", 610405], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "dbscan_lowd_self_q1500_m1500_d2_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 1500], ["M", 1500], ["D", 2], ["K", 64], ["dtype", "bfloat16"], ["seed", 610407], ["self_search", true], ["min_recall", 0.999]]}]]}, {"__dict_items__": [["label", "ksweep_q4096_m20000_d128_k64"], ["params", {"__dict_items__": [["B", 1], ["Q", 4096], ["M", 20000], ["D", 128], ["K", 64], ["dtype", "bfloat16"], ["seed", 610313], ["self_search", false], ["min_recall", 0.999]]}]]}]'))
SHAPE_DISPATCH_REGISTRY: tuple[dict[str, str], ...] = ({'shape_key': 'd128_q128_small_mid_m_k10', 'guard': 'B == 1 and Q == 128 and D == 128 and 8192 <= M <= 65536 and K <= 10 and tcgen05', 'route': 'round37_d128_q128_split_policy_small_mid'}, {'shape_key': 'd128_q128_large_m_lowk', 'guard': 'B == 1 and Q == 128 and D == 128 and M >= 131072 and K in {1,2,5,8,10} and tcgen05', 'route': 'round37_d128_q128_split_policy_large_m_lowk'}, *parent.parent.SHAPE_DISPATCH_REGISTRY, {'shape_key': 'inherited_r37_guard_miss', 'guard': 'otherwise', 'route': 'round37_parent_dispatch'})

def _use_q128_small_mid_policy(inputs: dict[str, Any]) -> bool:
    m_rows = int(inputs['M'])
    return parent._use_d128_q128_policy(inputs) and 8192 <= m_rows <= 65536

def _use_q128_large_lowk_policy(inputs: dict[str, Any]) -> bool:
    return parent._use_d128_q128_policy(inputs) and int(inputs['M']) >= 131072 and (int(inputs['K']) in Q128_LOWK_VALUES)

def selected_route(inputs: dict[str, Any]) -> str:
    if _use_q128_small_mid_policy(inputs):
        return 'round37_d128_q128_split_policy_small_mid'
    if _use_q128_large_lowk_policy(inputs):
        return 'round37_d128_q128_split_policy_large_m_lowk'
    return parent.selected_route(inputs)

def launch_for_eval(inputs: dict[str, Any]) -> dict[str, Any]:
    return parent.launch_for_eval(inputs)

def _select_contract_shapes(shape_labels: str | tuple[str, ...] | list[str] | None):
    if shape_labels is None:
        return None
    return select_named_shapes(shape_labels)

def knn_search_compile_and_launch_lowk_dispatch(*, benchmark: bool=True, shapes: list[dict[str, Any]] | None=None) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND38_DISPATCHER_SHAPES if shapes is None else shapes, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result

def knn_search_compile_and_launch_lowk_preserve(*, benchmark: bool=True) -> dict[str, Any]:
    from .._dispatch_runtime import evaluate
    result = evaluate(launch_for_eval, shapes=ROUND38_PRESERVE_SHAPES, benchmark=benchmark)
    result['passed'] = bool(result.get('summary', {}).get('all_correct'))
    print(result)
    return result
